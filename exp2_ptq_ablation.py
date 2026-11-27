#!/usr/bin/env python3
"""
Experimento 2: de onde vem a piora da deteccao de desconhecidas na PTQ, e como corrigir.

Reaproveita os checkpoints FP32 e as divisoes do Experimento 1 (nada e treinado de novo).
Cada configuracao abaixo e uma PTQ diferente do mesmo modelo FP32, avaliada com os mesmos
5 detectores. Linhas vao para <out>/results_ablation.csv com variant = abl_<config>.

Configuracoes (--configs, padrao: todas):
  base        o padrao do backend. No Mac/Pi (qnnpack): pesos por tensor, ativacoes com
              HistogramObserver. Deve reproduzir a ptq_int8 do Experimento 1.
  pc          pesos POR CANAL (ativacoes iguais a base)
  minmax      ativacoes com MinMaxObserver (pesos como na base)
  pc_minmax   por canal + MinMax
  w_only_pt   so os pesos quantizados (simulado, por tensor), ativacoes em FP32
  w_only_pc   so os pesos quantizados (simulado, por canal), ativacoes em FP32
  fc32        base, mas com a ultima camada linear em FP32
  cls32       base, mas com o classificador inteiro em FP32 (MobileNets)
  last1       base, com o ultimo bloco de features + classificador em FP32 (MobileNets)
  last3       base, com os 3 ultimos blocos + classificador em FP32 (MobileNets)
  pc_fc32     por canal + ultima camada em FP32
  cal25       base, calibracao com 25 imagens
  cal400      base, calibracao com 400 imagens

Isolamento das camadas depthwise (convolucoes com groups == canais de entrada):
  w_only_dw    so os pesos das depthwise quantizados (simulado, por tensor), o resto em FP32
  w_only_nodw  os pesos de todas as camadas, MENOS as depthwise (simulado, por tensor)
  dw_pt        INT8 completo: por tensor so nas depthwise, por canal no resto
  dw_pc        INT8 completo: por canal so nas depthwise, por tensor no resto (padrao)

Sensibilidade por tipo de camada (so os pesos daquele tipo, por tensor, o resto em FP32):
  w_only_stem  convolucao 3x3 de entrada
  w_only_pw    convolucoes pointwise 1x1 de expansao e de projecao dos blocos
  w_only_se    convolucoes 1x1 do squeeze-and-excitation (so MobileNetV3)
  w_only_head  ultima convolucao 1x1 e camadas lineares do classificador
  w_only_none  nenhum peso quantizado: o FP32 avaliado no mesmo dispositivo (referencia para --fp32-device)
  (as depthwise sao a w_only_dw acima)

Tambem mede, com --layer-error, o erro relativo de quantizacao bloco a bloco
(||INT8 - FP32|| / ||FP32||) em imagens conhecidas e desconhecidas: layer_error.csv.

Uso:
  python exp2_ptq_ablation.py --dataset-name macro --out results_macro --seeds 0 1 2 3 4 5 6 7 8 9
"""

import argparse
import copy
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
from torch.ao.quantization import (DeQuantStub, HistogramObserver, MinMaxObserver, QConfig, convert,
                                   default_per_channel_weight_observer, default_weight_observer,
                                   get_default_qconfig, prepare)

import fsrood as F
import run_exp1 as R

CONFIGS = ["base", "pc", "minmax", "pc_minmax", "w_only_pt", "w_only_pc", "fc32", "cls32",
           "last1", "last3", "pc_fc32", "cal25", "cal400",
           "w_only_dw", "w_only_nodw", "dw_pt", "dw_pc",
           "w_only_stem", "w_only_pw", "w_only_se", "w_only_head", "w_only_none"]
DW_CONFIGS = ["w_only_dw", "w_only_nodw", "dw_pt", "dw_pc"]
TYPE_CONFIGS = ["w_only_dw", "w_only_nodw", "w_only_stem", "w_only_pw", "w_only_se", "w_only_head"]
FP32_CONFIGS = TYPE_CONFIGS + ["w_only_pt", "w_only_pc", "w_only_none"]   # modelos FP32 (pesos simulados)


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../base")
    ap.add_argument("--layout", default="auto", choices=["auto", "filename", "folders"])
    ap.add_argument("--class-digits", type=int, default=2)
    ap.add_argument("--groups", default=None, help="o mesmo CSV de grupos usado no Experimento 1")
    ap.add_argument("--dataset-name", required=True)
    ap.add_argument("--out", required=True, help="pasta do Experimento 1 (com ckpt/ e splits/)")
    ap.add_argument("--backbones", nargs="+", default=["mobilenet_v3_large"], choices=F.BACKBONES)
    ap.add_argument("--seeds", nargs="+", type=int, default=list(range(10)))
    ap.add_argument("--configs", nargs="+", default=CONFIGS, choices=CONFIGS)
    ap.add_argument("--short-side", type=int, default=672)
    ap.add_argument("--max-patches", type=int, default=12)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--qengine", default="auto")
    ap.add_argument("--calib-images", type=int, default=100)
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--fp32-device", default="cpu",
                    help="dispositivo para as configuracoes so de pesos (modelos FP32): cpu, mps ou auto. "
                         "Fora da CPU a latencia nao e medida. Compare sempre com a w_only_none do mesmo dispositivo.")
    ap.add_argument("--save-scores", action="store_true")
    ap.add_argument("--save-feats", action="store_true")
    ap.add_argument("--layer-error", action="store_true", help="mede o erro bloco a bloco (base e pc)")
    ap.add_argument("--layer-error-images", type=int, default=40, help="imagens de cada grupo")
    return ap.parse_args()


# ---------------------------------------------------------------------------
# Construcao das variantes
# ---------------------------------------------------------------------------

def qconfig_for(cfg, engine):
    base = get_default_qconfig(engine)
    per_channel = cfg.startswith("pc")
    minmax = "minmax" in cfg
    act = MinMaxObserver.with_args(reduce_range=(engine != "qnnpack")) if minmax else base.activation
    # sem "pc": o observador de pesos padrao do backend (por tensor no qnnpack, usado no Mac e no
    # Raspberry Pi; ja por canal no x86, onde base e pc coincidem)
    w = default_per_channel_weight_observer if per_channel else base.weight
    return QConfig(activation=act, weight=w)


def _float_region(model, name, cfg):
    """Marca como FP32 (qconfig None) a parte final do modelo, com um DeQuantStub na fronteira."""
    if cfg in ("fc32", "pc_fc32"):
        head = F.head_of(model, name)
        head.qconfig = None
        wrapped = nn.Sequential(DeQuantStub(), head)
        if name == "mobilenet_v3_large":
            model.classifier[3] = wrapped
        elif name == "mobilenet_v2":
            model.classifier[1] = wrapped
        else:
            model.fc = wrapped
        return
    if not name.startswith("mobilenet"):
        raise ValueError(f"{cfg} so esta implementado para MobileNets")
    feats = model.features
    k = {"cls32": len(feats), "last1": len(feats) - 1, "last3": len(feats) - 3}[cfg]
    if k < len(feats):
        for i in range(k, len(feats)):
            feats[i].qconfig = None
        feats[k] = nn.Sequential(DeQuantStub(), feats[k])
        model.classifier.qconfig = None
    else:
        model.classifier.qconfig = None
        model.classifier = nn.Sequential(DeQuantStub(), model.classifier)
    if hasattr(model, "avgpool"):
        model.avgpool.qconfig = None


def is_depthwise(mod):
    """Convolucao depthwise: um filtro por canal de entrada (groups == in_channels > 1)."""
    if isinstance(mod, nni._FusedModule):
        mod = mod[0]
    return isinstance(mod, nn.Conv2d) and mod.groups > 1 and mod.groups == mod.in_channels


def layer_type(qname, mod, n_features):
    """Tipo de uma camada com pesos das MobileNets: stem, dw, se, head ou pw."""
    if isinstance(mod, nn.Linear):
        return "head"
    if mod.in_channels == 3:
        return "stem"
    if is_depthwise(mod):
        return "dw"
    if ".fc1" in qname or ".fc2" in qname:
        return "se"
    if qname.startswith(f"features.{n_features - 1}."):
        return "head"
    return "pw"


@torch.no_grad()
def _fake_quant_weights(model, per_channel, only=None):
    """Quantiza e desquantiza so os pesos (INT8 simetrico), mantendo ativacoes em FP32.
    only: None (todas as camadas), "nodw" (todas menos as depthwise) ou um tipo de layer_type
    ("dw", "stem", "pw", "se", "head")."""
    nf = len(model.features) if hasattr(model, "features") else 0
    for qname, mod in model.named_modules():
        w = getattr(mod, "weight", None)
        if not isinstance(mod, (nn.Conv2d, nn.Linear)) or w is None:
            continue
        if only == "nodw":
            if is_depthwise(mod):
                continue
        elif only is not None and layer_type(qname, mod, nf) != only:
            continue
        if per_channel:
            amax = w.abs().flatten(1).amax(1).clamp(min=1e-12)
            scale = (amax / 127).view(-1, *([1] * (w.dim() - 1)))
        else:
            scale = (w.abs().max().clamp(min=1e-12) / 127)
        mod.weight.copy_(torch.clamp(torch.round(w / scale), -127, 127) * scale)


@torch.no_grad()
def build_variant(fp32, name, cfg, calib, cache_map, args, engine):
    m = copy.deepcopy(fp32).cpu().eval()
    m.fuse_model(is_qat=False)
    if cfg == "w_only_none":
        return m
    if cfg in TYPE_CONFIGS:
        _fake_quant_weights(m, per_channel=False, only=cfg.split("_")[-1])
        return m
    if cfg.startswith("w_only"):
        _fake_quant_weights(m, per_channel=cfg.endswith("pc"))
        return m
    if cfg in ("dw_pt", "dw_pc"):
        # o qconfig do modulo (fundido ou nao) define o observador dos pesos daquela camada
        rest, dw = ("pc", "base") if cfg == "dw_pt" else ("base", "pc")
        m.qconfig = qconfig_for(rest, engine)
        dwq = qconfig_for(dw, engine)
        for mod in m.modules():
            if is_depthwise(mod):
                mod.qconfig = dwq
                if isinstance(mod, nni._FusedModule):
                    mod[0].qconfig = dwq
    else:
        m.qconfig = qconfig_for(cfg, engine)
    if cfg in ("fc32", "pc_fc32", "cls32", "last1", "last3"):
        _float_region(m, name, cfg)
    prepare(m, inplace=True)
    for p, _ in calib:
        m(F.load_patches(cache_map[p], args.max_patches))
    convert(m, inplace=True)
    return m.eval()


# ---------------------------------------------------------------------------
# Erro bloco a bloco
# ---------------------------------------------------------------------------

def blocks_of(model, name):
    if name.startswith("mobilenet"):
        return [(f"features.{i}", b) for i, b in enumerate(model.features)] + [("logits", None)]
    return [(n, getattr(model, n)) for n in ("layer1", "layer2", "layer3", "layer4")] + [("logits", None)]


@torch.no_grad()
def layer_error(fp32, qm, name, items, cache_map, args):
    def collect(model):
        outs = {}
        hooks = []
        for n, b in blocks_of(model, name):
            if b is None:
                continue
            def h(_m, _i, o, n=n):
                o = o.dequantize() if o.is_quantized else o
                outs[n] = o.float()
            hooks.append(b.register_forward_hook(h))
        return outs, hooks

    rows = []
    of, hf = collect(fp32)
    oq, hq = collect(qm)
    try:
        for p, _ in items:
            x = F.load_patches(cache_map[p], args.max_patches)
            lf, lq = fp32(x), qm(x)
            lq = lq.dequantize() if lq.is_quantized else lq
            errs = {n: float((oq[n] - of[n]).norm() / of[n].norm().clamp(min=1e-12)) for n in of}
            errs["logits"] = float((lq - lf).norm() / lf.norm().clamp(min=1e-12))
            rows.append(errs)
    finally:
        for h in hf + hq:
            h.remove()
    return rows


# ---------------------------------------------------------------------------

def main():
    args = parse()
    F.raise_open_files_limit()
    out = Path(args.out)
    engine = F.pick_qengine(args.qengine)
    fp32_dev = args.fp32_device
    if fp32_dev == "auto":
        fp32_dev = "mps" if torch.backends.mps.is_available() else "cpu"
    if fp32_dev == "mps" and not torch.backends.mps.is_available():
        print("[aviso] mps indisponivel, usando cpu")
        fp32_dev = "cpu"
    print(f"[info] torch {torch.__version__} | engine INT8 = {engine} | modelos FP32 em {fp32_dev}")
    classes, samples = F.discover(args.data, args.layout, args.class_digits)
    cache_map = F.build_cache(samples, args.data, args.cache or Path(args.data).resolve().parent / "cache",
                              args.short_side)
    groups = F.load_groups(args.groups, args.data, samples) if args.groups else None
    csv_path = out / "results_ablation.csv"
    done = R.done_keys(csv_path)
    le_path = out / "layer_error.csv"
    le_done = set()
    if le_path.exists():
        with open(le_path) as f:
            le_done = {(r["dataset"], int(r["seed"]), r["backbone"], r["config"]) for r in csv.DictReader(f)}

    for seed in args.seeds:
        split = F.make_split(classes, samples, seed, 0.5, groups=groups)
        sp = out / "splits" / f"{args.dataset_name}_seed{seed}.json"
        if sp.exists():
            import json
            saved = json.load(open(sp))
            if saved["known_classes"] != split["known_classes"]:
                raise RuntimeError(f"divisao da semente {seed} difere da gravada em {sp}")
        nk, nu = len(split["known_classes"]), len(split["unknown_classes"])
        rng = np.random.RandomState(seed)
        # RandomState.choice(sem reposicao) = permutation()[:n]: as 100 primeiras sao exatamente
        # as imagens de calibracao da ptq_int8 do Experimento 1
        perm = rng.permutation(len(split["train"]))
        for name in args.backbones:
            ckpt = out / "ckpt" / f"{args.dataset_name}_{name}_seed{seed}_fp32.pt"
            if not ckpt.exists():
                print(f"[skip] sem checkpoint {ckpt.name} (rode o Experimento 1 antes)")
                continue
            if name in F.FX_MODELS:
                print(f"[skip] {name}: ablacao so para modelos quantizaveis do torchvision "
                      f"(use a variante ptq_int8_pc do run_exp1.py)")
                continue
            fp32 = F.build_model(name, nk, pretrained=False)
            fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
            fp32.eval()
            ctx = dict(dataset=args.dataset_name, seed=seed, backbone=name, n_known=nk, n_unknown=nu,
                       n_test_known=len(split["test_known"]), n_test_unknown=len(split["test_unknown"]),
                       qengine=engine, torch=torch.__version__)
            for cfg in args.configs:
                if cfg in ("cls32", "last1", "last3") + tuple(TYPE_CONFIGS) and not name.startswith("mobilenet"):
                    continue
                if cfg == "w_only_se" and name != "mobilenet_v3_large":
                    continue
                v = f"abl_{cfg}"
                if (args.dataset_name, seed, name, v) in done:
                    continue
                n_cal = {"cal25": 25, "cal400": 400}.get(cfg, args.calib_images)
                calib = [split["train"][i] for i in perm[:min(n_cal, len(perm))]]
                t0 = time.time()
                qm = build_variant(fp32, name, cfg, calib, cache_map, args, engine)
                dev = fp32_dev if cfg in FP32_CONFIGS else "cpu"
                rows = R.evaluate(qm, name, split, cache_map, args, dict(ctx, variant=v),
                                  device=dev)
                R.append_rows(csv_path, rows)
                byd = {r["detector"]: r for r in rows}
                print(f"[{v}] {name} s{seed}: acc={rows[0]['acc']:.4f} | FPR95 energy={byd['energy']['fpr95']:.3f} "
                      f"knn={byd['knn']['fpr95']:.3f} | {rows[0]['size_mb']:.1f} MB | {dev} | {time.time() - t0:.0f}s")

                if args.layer_error and cfg in ("base", "pc") and (args.dataset_name, seed, name, cfg) not in le_done:
                    n = args.layer_error_images
                    rs = np.random.RandomState(seed + 1000)
                    sel_k = [split["test_known"][i] for i in rs.choice(len(split["test_known"]),
                             min(n, len(split["test_known"])), replace=False)]
                    sel_u = [split["test_unknown"][i] for i in rs.choice(len(split["test_unknown"]),
                             min(n, len(split["test_unknown"])), replace=False)]
                    fq = copy.deepcopy(fp32).eval()
                    fq.fuse_model(is_qat=False)          # mesma estrutura de blocos do modelo INT8
                    le_rows = []
                    for grp, items in (("known", sel_k), ("unknown", sel_u)):
                        errs = layer_error(fq, qm, name, items, cache_map, args)
                        for blk in errs[0]:
                            vals = [e[blk] for e in errs]
                            le_rows.append(dict(dataset=args.dataset_name, seed=seed, backbone=name, config=cfg,
                                                group=grp, block=blk, rel_err_mean=float(np.mean(vals)),
                                                rel_err_std=float(np.std(vals)), n_images=len(vals)))
                    new = not le_path.exists()
                    with open(le_path, "a", newline="") as f:
                        w = csv.DictWriter(f, fieldnames=list(le_rows[0]))
                        if new:
                            w.writeheader()
                        w.writerows(le_rows)
                    print(f"  [layer_error] {cfg}: logits known={[r for r in le_rows if r['block']=='logits' and r['group']=='known'][0]['rel_err_mean']:.3f} "
                          f"unknown={[r for r in le_rows if r['block']=='logits' and r['group']=='unknown'][0]['rel_err_mean']:.3f}")
    print(f"\n[ok] {csv_path}")


if __name__ == "__main__":
    main()
