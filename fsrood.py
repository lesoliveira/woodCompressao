"""
fsrood.py - biblioteca do Experimento 1 (compressao x deteccao de especies desconhecidas).

Contem: descoberta da base, divisao conhecidas/desconhecidas, cache de imagens
redimensionadas, extracao de patches, construcao dos modelos quantizaveis
(torchvision), treino FP32, PTQ e QAT em INT8 (modo eager, kernels INT8 reais),
inferencia por imagem (media dos patches), detectores pos-hoc e metricas.
"""

import copy
import io
import json
import math
import os
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
PATCH = 224


# ---------------------------------------------------------------------------
# Reprodutibilidade e dispositivo
# ---------------------------------------------------------------------------

def raise_open_files_limit(target=16384):
    """O macOS limita a 256 arquivos abertos por processo. Os workers do DataLoader (spawn) usam
    pipes e descritores, e um processo que treina varios modelos seguidos estoura esse limite
    (OSError: [Errno 24] Too many open files). Sobe o limite ate o maximo permitido."""
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        want = target if hard == resource.RLIM_INFINITY else min(target, hard)
        if soft < want:
            resource.setrlimit(resource.RLIMIT_NOFILE, (want, hard))
    except (ImportError, ValueError, OSError):
        pass


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_train_device(pref="auto"):
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def pick_qengine(pref="auto"):
    """qnnpack em ARM (Mac M*, Raspberry Pi, celular), x86/fbgemm em Intel/AMD."""
    if pref != "auto":
        eng = pref
    elif platform.machine().lower() in ("arm64", "aarch64", "armv7l"):
        eng = "qnnpack"
    else:
        eng = "x86" if "x86" in torch.backends.quantized.supported_engines else "fbgemm"
    torch.backends.quantized.engine = eng
    return eng


# ---------------------------------------------------------------------------
# Base de dados
# ---------------------------------------------------------------------------

def discover(root, layout="auto", class_digits=2):
    """Retorna (lista de classes, lista de (caminho, idx_classe)).

    layout="filename": todas as imagens numa pasta so, e os primeiros `class_digits`
        caracteres do nome dao a classe (base macroscopica: 0101.JPG = classe 01, imagem 01).
    layout="folders":  root/<especie>/<imagem>, ou aninhado (root/Hardwood/<especie>/<imagem>):
        cada pasta que contem imagens vira uma classe, com nome = caminho relativo.
    layout="auto":     "folders" se houver subpastas com imagens, senao "filename".
    """
    root = Path(root)
    is_img = lambda p: p.is_file() and p.suffix.lower() in IMG_EXTS and not p.name.startswith(".")
    subdirs = sorted(d for d in root.iterdir() if d.is_dir() and not d.name.startswith("."))
    if layout == "auto":
        layout = "folders" if any(any(is_img(p) for p in d.rglob("*")) for d in subdirs) else "filename"

    if layout == "folders":
        # cada pasta que contem imagens e uma classe (aceita aninhamento, ex.: Hardwood/<especie>/)
        dirs = sorted({p.parent for p in root.rglob("*") if is_img(p)})
        classes = [str(d.relative_to(root)) for d in dirs]
        idx = {d: i for i, d in enumerate(dirs)}
        samples = [(str(p), idx[p.parent]) for p in sorted(root.rglob("*")) if is_img(p)]
    elif layout == "filename":
        files = sorted(p for p in root.iterdir() if is_img(p))
        bad = [p.name for p in files if not p.stem[:class_digits].isdigit()]
        if bad:
            raise RuntimeError(f"Nomes fora do padrao (esperado {class_digits} digitos de classe no inicio): {bad[:5]}")
        classes = sorted({p.stem[:class_digits] for p in files})
        idx = {c: i for i, c in enumerate(classes)}
        samples = [(str(p), idx[p.stem[:class_digits]]) for p in files]
    else:
        raise ValueError(f"layout invalido: {layout}")

    if not classes or not samples:
        raise RuntimeError(f"Nenhuma imagem encontrada em {root}")
    return classes, samples


def load_groups(csv_path, root, samples):
    """Le um CSV file,group (file relativo a raiz da base) e devolve {caminho da amostra: grupo}.
    Imagens do mesmo grupo (repetidas ou com campos sobrepostos) ficam sempre na mesma particao."""
    import csv as _csv
    root = Path(root)
    with open(csv_path) as f:
        g = {r["file"]: r["group"] for r in _csv.DictReader(f)}
    out, miss = {}, 0
    for p, _ in samples:
        rel = str(Path(p).relative_to(root)) if Path(p).is_relative_to(root) else Path(p).name
        if rel in g:
            out[p] = g[rel]
        else:
            out[p] = f"_single_{p}"
            miss += 1
    if miss:
        print(f"[aviso] {miss} imagens sem grupo no CSV: tratadas como grupos unitarios")
    return out


def make_split(classes, samples, seed, known_frac=0.5, fracs=(0.6, 0.2, 0.2), groups=None):
    """Divide especies em conhecidas/desconhecidas e as imagens das conhecidas em treino/val/teste.
    Com groups ({caminho: grupo}), a divisao das conhecidas e feita por grupo, nunca separando
    imagens do mesmo grupo. Sem groups, o resultado e identico ao das versoes anteriores."""
    rng = random.Random(seed)
    cls_idx = list(range(len(classes)))
    rng.shuffle(cls_idx)
    n_known = max(2, round(len(classes) * known_frac))
    known = sorted(cls_idx[:n_known])
    unknown = sorted(cls_idx[n_known:])
    remap = {c: i for i, c in enumerate(known)}

    by_class = {}
    for p, c in samples:
        by_class.setdefault(c, []).append(p)

    split = {"train": [], "val": [], "test_known": [], "test_unknown": []}
    for c in known:
        imgs = sorted(by_class[c])
        if groups is not None:
            gmap = {}
            for p in imgs:
                gmap.setdefault(groups[p], []).append(p)
            glist = [gmap[k] for k in sorted(gmap)]
            rng.shuffle(glist)
            n = len(imgs)
            parts, acc = ([], [], []), 0
            for g in glist:
                k = 0 if acc < round(n * fracs[0]) else 1 if acc < round(n * (fracs[0] + fracs[1])) else 2
                parts[k].extend(g)
                acc += len(g)
            if not parts[2] and len(parts[1]) > 1:      # garante teste nao vazio
                parts[2].append(parts[1].pop())
            split["train"] += [(p, remap[c]) for p in parts[0]]
            split["val"] += [(p, remap[c]) for p in parts[1]]
            split["test_known"] += [(p, remap[c]) for p in parts[2]]
            continue
        rng.shuffle(imgs)
        n = len(imgs)
        n_tr = max(1, int(round(n * fracs[0])))
        n_va = max(1, int(round(n * fracs[1])))
        if n_tr + n_va >= n:
            n_va = max(0, n - n_tr - 1)
        split["train"] += [(p, remap[c]) for p in imgs[:n_tr]]
        split["val"] += [(p, remap[c]) for p in imgs[n_tr:n_tr + n_va]]
        split["test_known"] += [(p, remap[c]) for p in imgs[n_tr + n_va:]]
    for c in unknown:
        split["test_unknown"] += [(p, -1) for p in by_class[c]]

    split["known_classes"] = [classes[c] for c in known]
    split["unknown_classes"] = [classes[c] for c in unknown]
    split["seed"] = seed
    split["grouped"] = groups is not None
    return split


def build_cache(samples, root, cache_dir, short_side):
    """Grava copias redimensionadas (lado menor = short_side). Retorna mapa original -> cache."""
    root = Path(root).resolve()
    cache_dir = Path(cache_dir) / f"short{short_side}"
    mapping = {}
    todo = 0
    for p, _ in samples:
        rel = Path(p).resolve().relative_to(root)
        out = (cache_dir / rel).with_suffix(".jpg")
        mapping[p] = str(out)
        if not out.exists():
            todo += 1
    if todo:
        print(f"[cache] redimensionando {todo} imagens para lado menor {short_side} em {cache_dir}")
    done = 0
    for p, out in mapping.items():
        out = Path(out)
        if out.exists():
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(p) as f:
            im = f.convert("RGB")
        w, h = im.size
        s = short_side / min(w, h)
        im = im.resize((max(PATCH, round(w * s)), max(PATCH, round(h * s))), Image.BICUBIC)
        im.save(out, quality=95)
        done += 1
        if done % 200 == 0:
            print(f"[cache] {done}/{todo}")
    return mapping


def grid_boxes(w, h, max_patches=0):
    """Patches 224x224 nao sobrepostos, em grade centrada na imagem."""
    nx, ny = max(1, w // PATCH), max(1, h // PATCH)
    x0, y0 = (w - nx * PATCH) // 2, (h - ny * PATCH) // 2
    boxes = [(x0 + i * PATCH, y0 + j * PATCH, x0 + (i + 1) * PATCH, y0 + (j + 1) * PATCH)
             for j in range(ny) for i in range(nx)]
    if max_patches and len(boxes) > max_patches:
        idx = np.linspace(0, len(boxes) - 1, max_patches).round().astype(int)
        boxes = [boxes[i] for i in idx]
    return boxes


NORM = T.Compose([T.ToTensor(), T.Normalize(MEAN, STD)])


class TrainCrops(Dataset):
    """Cada imagem gera crops_per_image recortes aleatorios por epoca."""

    def __init__(self, items, cache_map, crops_per_image=4):
        self.items = items
        self.cache = cache_map
        self.k = crops_per_image
        self.aug = T.Compose([
            T.RandomCrop(PATCH),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.ColorJitter(0.1, 0.1, 0.05),
        ])

    def __len__(self):
        return len(self.items) * self.k

    def __getitem__(self, i):
        p, y = self.items[i // self.k]
        with Image.open(self.cache[p]) as f:
            im = f.convert("RGB")
        return NORM(self.aug(im)), y


def load_patches(path, max_patches=0):
    with Image.open(path) as f:
        im = f.convert("RGB")
    return torch.stack([NORM(im.crop(b)) for b in grid_boxes(*im.size, max_patches)])


# ---------------------------------------------------------------------------
# Modelos (versoes quantizaveis do torchvision)
# ---------------------------------------------------------------------------

BACKBONES = ("mobilenet_v3_large", "mobilenet_v2", "resnet18")


def build_model(name, n_classes, pretrained=True):
    from torchvision.models import quantization as Q
    from torchvision import models as M
    if name == "mobilenet_v3_large":
        w = M.MobileNet_V3_Large_Weights.IMAGENET1K_V1 if pretrained else None
        m = Q.mobilenet_v3_large(weights=w, quantize=False)
        m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    elif name == "mobilenet_v2":
        w = M.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
        m = Q.mobilenet_v2(weights=w, quantize=False)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, n_classes)
    elif name == "resnet18":
        w = M.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        m = Q.resnet18(weights=w, quantize=False)
        m.fc = nn.Linear(m.fc.in_features, n_classes)
    else:
        raise ValueError(f"backbone desconhecido: {name} (opcoes: {BACKBONES})")
    return m


def head_of(model, name):
    """Ultima camada linear (float ou quantizada). A entrada dela e a feature penultima usada
    por kNN/Mahalanobis. Procura pelo tipo, para funcionar tambem em modelos com partes em FP32."""
    import torch.ao.nn.quantized as nnq
    last = None
    for mod in model.modules():
        if isinstance(mod, (nn.Linear, nnq.Linear)):
            last = mod
    if last is None:
        raise RuntimeError("camada linear final nao encontrada")
    return last


def model_size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2 ** 20


# ---------------------------------------------------------------------------
# Treino
# ---------------------------------------------------------------------------

def _loader(items, cache_map, args, shuffle=True):
    ds = TrainCrops(items, cache_map, args.crops_per_image)
    return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=args.workers, drop_last=len(ds) > args.batch_size)


def _one_epoch(model, loader, opt, device, sched=None, freeze_bn=False):
    model.train()
    if freeze_bn:
        _freeze_bn(model)
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        loss = F.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if sched is not None:
            sched.step()
        tot += loss.item() * len(y)
        n += len(y)
    return tot / max(n, 1)


def train_fp32(name, n_classes, split, cache_map, args, device):
    model = build_model(name, n_classes, pretrained=not args.no_pretrained).to(device)
    loader = _loader(split["train"], cache_map, args)
    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9,
                          weight_decay=args.weight_decay, nesterov=True)
    steps = args.epochs * len(loader)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=max(steps, 1),
                                                pct_start=0.1)
    best, best_state = -1.0, None
    for ep in range(args.epochs):
        t0 = time.time()
        loss = _one_epoch(model, loader, opt, device, sched)
        acc = closed_accuracy(model, name, split["val"], cache_map, args, device) if split["val"] else 0.0
        print(f"  [fp32 {name}] epoca {ep + 1}/{args.epochs} loss={loss:.4f} val_acc={acc:.4f} "
              f"({time.time() - t0:.0f}s)")
        if acc >= best:
            best, best_state = acc, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model.cpu().eval(), best


@torch.no_grad()
def closed_accuracy(model, name, items, cache_map, args, device):
    out = infer(model, name, items, cache_map, args, device, need_feats=False)
    return float((out["logits"].argmax(1) == out["labels"]).mean())


# ---------------------------------------------------------------------------
# Quantizacao (eager mode, kernels INT8 reais do backend escolhido)
# ---------------------------------------------------------------------------

from torch.ao.quantization import (convert, get_default_qat_qconfig, get_default_qconfig,
                                   prepare, prepare_qat)


@torch.no_grad()
def ptq_int8(fp32_model, calib_items, cache_map, args, engine):
    m = copy.deepcopy(fp32_model).cpu().eval()
    m.fuse_model(is_qat=False)
    m.qconfig = get_default_qconfig(engine)
    prepare(m, inplace=True)
    for p, _ in calib_items:
        m(load_patches(cache_map[p], args.max_patches))
    convert(m, inplace=True)
    return m.eval()


def finetune_fp32(fp32_model, split, cache_map, args):
    """Controle da QAT: o MESMO ajuste fino (epocas, lr, otimizador, congelamento de BN,
    CPU), mas sem quantizacao. Separa o efeito do ajuste fino extra do efeito da quantizacao."""
    m = copy.deepcopy(fp32_model).cpu()
    m.train()
    loader = _loader(split["train"], cache_map, args)
    opt = torch.optim.SGD(m.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=args.weight_decay)
    for ep in range(args.qat_epochs):
        t0 = time.time()
        # mesma regra da QAT: estatisticas de BN congeladas na ultima epoca (se houver mais de uma)
        freeze = args.qat_epochs > 1 and ep == max(1, args.qat_epochs - 1)
        loss = _one_epoch(m, loader, opt, torch.device("cpu"), freeze_bn=freeze)
        print(f"  [fp32_ft] epoca {ep + 1}/{args.qat_epochs} loss={loss:.4f} ({time.time() - t0:.0f}s)")
    return m.eval()


def _freeze_bn(m):
    for mod in m.modules():
        if isinstance(mod, nn.modules.batchnorm._BatchNorm):
            mod.eval()


def qat_int8(fp32_model, split, cache_map, args, engine, per_channel=False):
    """QAT a partir do FP32. per_channel=True troca o fake-quant dos pesos por um por canal
    (no qnnpack, usado no Mac e no Raspberry Pi, o padrao e por tensor; no x86 ja e por canal)."""
    import torch.ao.nn.intrinsic.qat as nniqat
    from torch.ao.quantization import QConfig, default_per_channel_weight_fake_quant
    m = copy.deepcopy(fp32_model).cpu()
    m.train()
    m.fuse_model(is_qat=True)
    qc = get_default_qat_qconfig(engine)
    if per_channel:
        qc = QConfig(activation=qc.activation, weight=default_per_channel_weight_fake_quant)
    m.qconfig = qc
    prepare_qat(m, inplace=True)
    loader = _loader(split["train"], cache_map, args)
    opt = torch.optim.SGD(m.parameters(), lr=args.qat_lr, momentum=0.9, weight_decay=args.weight_decay)
    for ep in range(args.qat_epochs):
        if ep == max(1, args.qat_epochs - 1):
            m.apply(nniqat.freeze_bn_stats)          # congela estatisticas de BN na ultima epoca
        if ep == args.qat_epochs - 1 and args.qat_epochs > 1:
            m.apply(torch.ao.quantization.disable_observer)
        t0 = time.time()
        loss = _one_epoch(m, loader, opt, torch.device("cpu"))
        print(f"  [qat] epoca {ep + 1}/{args.qat_epochs} loss={loss:.4f} ({time.time() - t0:.0f}s)")
    m.eval()
    return convert(m, inplace=False).eval()


# ---------------------------------------------------------------------------
# Inferencia por imagem (media de logits e features dos patches)
# ---------------------------------------------------------------------------

@torch.no_grad()
def infer(model, name, items, cache_map, args, device=torch.device("cpu"), need_feats=True):
    model.eval()
    feats_buf = []
    hook = None
    if need_feats:
        def pre_hook(_m, inp):
            x = inp[0]
            feats_buf.append(x.dequantize() if x.is_quantized else x)
        hook = head_of(model, name).register_forward_pre_hook(pre_hook)
    logits, feats, labels = [], [], []
    try:
        for p, y in items:
            x = load_patches(cache_map[p], args.max_patches).to(device)
            feats_buf.clear()
            lg = model(x)
            lg = lg.dequantize() if lg.is_quantized else lg
            logits.append(lg.float().mean(0).cpu())
            if need_feats:
                feats.append(torch.flatten(feats_buf[0], 1).float().mean(0).cpu())
            labels.append(y)
    finally:
        if hook is not None:
            hook.remove()
    out = {"logits": torch.stack(logits).numpy(), "labels": np.array(labels)}
    if need_feats:
        out["feats"] = torch.stack(feats).numpy()
    return out


@torch.no_grad()
def latency_ms(model, items, cache_map, args, n=20):
    """Mediana do tempo por imagem (todos os patches), em CPU, com args.threads threads."""
    old = torch.get_num_threads()
    torch.set_num_threads(args.threads)
    try:
        xs = [load_patches(cache_map[p], args.max_patches) for p, _ in items[:n]]
        model(xs[0])  # aquecimento
        ts = []
        for x in xs:
            t0 = time.perf_counter()
            model(x)
            ts.append((time.perf_counter() - t0) * 1000)
        return float(np.median(ts))
    finally:
        torch.set_num_threads(old)


# ---------------------------------------------------------------------------
# Detectores pos-hoc (escore maior = mais "conhecida")
# ---------------------------------------------------------------------------

def _softmax(z):
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def _logsumexp(z):
    m = z.max(1, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(1, keepdims=True)))[:, 0]


def _l2n(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


class FeatureDetectors:
    """kNN (Sun et al., 2022) e Mahalanobis com covariancia compartilhada (Lee et al., 2018).
    Ajustados nas features de treino do PROPRIO modelo avaliado (FP32 ou INT8)."""

    def __init__(self, train_feats, train_labels, k=5):
        from sklearn.covariance import LedoitWolf
        self.k = min(k, len(train_feats))
        self.bank = _l2n(train_feats)
        classes = np.unique(train_labels)
        self.means = np.stack([train_feats[train_labels == c].mean(0) for c in classes])
        centered = train_feats - self.means[np.searchsorted(classes, train_labels)]
        lw = LedoitWolf().fit(centered)          # encolhimento: D (512-1280) > n de imagens
        self.prec = lw.precision_

    def knn(self, feats):
        sims = _l2n(feats) @ self.bank.T
        kth = np.sort(sims, axis=1)[:, -self.k]
        return -(np.sqrt(np.clip(2 - 2 * kth, 0, None)))   # -distancia ao k-esimo vizinho

    def mahalanobis(self, feats):
        d = feats[:, None, :] - self.means[None, :, :]
        md = np.einsum("ncd,de,nce->nc", d, self.prec, d)
        return -md.min(1)


DETECTORS = ("msp", "maxlogit", "energy", "knn", "mahalanobis")


def all_scores(out, fdet):
    z = out["logits"]
    return {
        "msp": _softmax(z).max(1),
        "maxlogit": z.max(1),
        "energy": _logsumexp(z),
        "knn": fdet.knn(out["feats"]),
        "mahalanobis": fdet.mahalanobis(out["feats"]),
    }


# ---------------------------------------------------------------------------
# Metricas
# ---------------------------------------------------------------------------

def auroc(s_known, s_unk):
    from sklearn.metrics import roc_auc_score
    y = np.r_[np.ones(len(s_known)), np.zeros(len(s_unk))]
    return float(roc_auc_score(y, np.r_[s_known, s_unk]))


def fpr_at_tpr(s_known, s_unk, tpr=0.95):
    thr = np.quantile(s_known, 1 - tpr)
    return float((s_unk >= thr).mean())


def oscr(s_known, correct, s_unk):
    """Open-Set Classification Rate (Dhamija et al., 2018): area sob CCR x FPR."""
    s = np.r_[s_known, s_unk]
    is_k = np.r_[np.ones(len(s_known), bool), np.zeros(len(s_unk), bool)]
    corr = np.r_[correct.astype(bool), np.zeros(len(s_unk), bool)]
    order = np.argsort(-s, kind="stable")
    ccr = np.r_[0, np.cumsum(corr[order] & is_k[order]) / max(len(s_known), 1)]
    fpr = np.r_[0, np.cumsum(~is_k[order]) / max(len(s_unk), 1)]
    return float(np.trapezoid(ccr, fpr) if hasattr(np, "trapezoid") else np.trapz(ccr, fpr))


def ece(logits, labels, n_bins=15):
    p = _softmax(logits)
    conf, pred = p.max(1), p.argmax(1)
    acc = pred == labels
    bins = np.linspace(0, 1, n_bins + 1)
    e = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.mean() * abs(acc[m].mean() - conf[m].mean())
    return float(e)


def save_json(obj, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
