#!/usr/bin/env python3
"""
Experimento 1: a quantizacao INT8 prejudica mais a deteccao de especies
desconhecidas do que a acuracia fechada?

Para cada semente e backbone:
  1. divide as especies em conhecidas/desconhecidas
  2. treina FP32 nas conhecidas (checkpoint reaproveitado se existir)
  3. gera INT8 por PTQ e por QAT (kernels INT8 reais, CPU)
  4. avalia FP32, PTQ e QAT com 5 detectores pos-hoc
  5. grava uma linha por (semente, backbone, variante, detector) em results.csv

Exemplo:
  python run_exp1.py --data ../base --dataset-name macro --out results_macro \
      --backbones mobilenet_v3_large resnet18 --seeds 0 1 2 3 4
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch

import fsrood as F

FIELDS = ["dataset", "seed", "backbone", "variant", "detector", "n_known", "n_unknown",
          "n_test_known", "n_test_unknown", "acc", "auroc", "fpr95", "oscr", "ece",
          "size_mb", "latency_ms", "qengine", "torch"]


def parse():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="../base",
                    help="pasta da base (padrao ../base). Imagens 0101.JPG: 2 digitos de classe + 2 de imagem")
    ap.add_argument("--layout", default="auto", choices=["auto", "filename", "folders"],
                    help="filename: classe nos primeiros digitos do nome | folders: uma pasta por especie")
    ap.add_argument("--class-digits", type=int, default=2, help="digitos de classe no inicio do nome")
    ap.add_argument("--groups", default=None,
                    help="CSV file,group: imagens do mesmo grupo (repetidas/sobrepostas) nunca se separam")
    ap.add_argument("--out", default="results_exp1")
    ap.add_argument("--dataset-name", default=None, help="rotulo no CSV (padrao: nome da pasta)")
    ap.add_argument("--backbones", nargs="+", default=["mobilenet_v3_large", "resnet18"],
                    choices=F.BACKBONES)
    ap.add_argument("--variants", nargs="+", default=["fp32", "ptq_int8", "qat_int8"],
                    choices=["fp32", "ptq_int8", "qat_int8", "fp32_ft"],
                    help="fp32_ft = controle da QAT: mesmo ajuste fino, sem quantizacao")
    ap.add_argument("--tag", default="",
                    help="sufixo no nome das variantes fp32_ft e qat_int8 no CSV (ex.: lr1e-4)")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--known-frac", type=float, default=0.5)
    # imagens e patches
    ap.add_argument("--short-side", type=int, default=672,
                    help="lado menor apos redimensionar (672 -> grade 3x4 de patches 224 em imagens 4:3)")
    ap.add_argument("--max-patches", type=int, default=12, help="0 = todos os patches da grade")
    ap.add_argument("--cache", default=None, help="pasta do cache redimensionado (padrao: <pasta da base>/../cache, compartilhado entre execucoes)")
    # treino FP32
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--crops-per-image", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--no-pretrained", action="store_true", help="so para testes rapidos")
    ap.add_argument("--device", default="auto", help="auto | cpu | mps | cuda (treino FP32)")
    ap.add_argument("--workers", type=int, default=2)
    # quantizacao
    ap.add_argument("--qengine", default="auto", help="auto | qnnpack | x86 | fbgemm")
    ap.add_argument("--calib-images", type=int, default=100, help="imagens de treino para calibrar a PTQ")
    ap.add_argument("--qat-epochs", type=int, default=3)
    ap.add_argument("--qat-lr", type=float, default=1e-3)
    # detectores e medicao
    ap.add_argument("--knn-k", type=int, default=5)
    ap.add_argument("--threads", type=int, default=4, help="threads de CPU na medida de latencia")
    ap.add_argument("--save-scores", action="store_true", help="grava escores por imagem (.npz)")
    ap.add_argument("--save-feats", action="store_true",
                    help="com --save-scores, grava tambem logits de treino e features penultimas (float16)")
    return ap.parse_args()


def done_keys(csv_path):
    if not csv_path.exists():
        return set()
    with open(csv_path) as f:
        return {(r["dataset"], int(r["seed"]), r["backbone"], r["variant"]) for r in csv.DictReader(f)}


def append_rows(csv_path, rows):
    new = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerows(rows)


def evaluate(model, name, split, cache_map, args, ctx):
    """Avalia um modelo (FP32 ou INT8) em CPU e devolve as linhas do CSV."""
    cpu = torch.device("cpu")
    tr = F.infer(model, name, split["train"], cache_map, args, cpu)
    tk = F.infer(model, name, split["test_known"], cache_map, args, cpu)
    tu = F.infer(model, name, split["test_unknown"], cache_map, args, cpu)
    fdet = F.FeatureDetectors(tr["feats"], tr["labels"], k=args.knn_k)
    sk, su = F.all_scores(tk, fdet), F.all_scores(tu, fdet)
    correct = tk["logits"].argmax(1) == tk["labels"]
    acc = float(correct.mean())
    e = F.ece(tk["logits"], tk["labels"])
    size = F.model_size_mb(model)
    lat = F.latency_ms(model, split["test_known"], cache_map, args)
    rows = []
    for d in F.DETECTORS:
        rows.append(dict(ctx, detector=d, acc=acc, auroc=F.auroc(sk[d], su[d]),
                         fpr95=F.fpr_at_tpr(sk[d], su[d]), oscr=F.oscr(sk[d], correct, su[d]),
                         ece=e, size_mb=size, latency_ms=lat))
    if args.save_scores:
        sdir = Path(args.out) / "scores"
        sdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(sdir / f"{ctx['dataset']}_s{ctx['seed']}_{name}_{ctx['variant']}.npz",
                            labels_known=tk["labels"], logits_known=tk["logits"],
                            logits_unknown=tu["logits"],
                            **({"feats_train": tr["feats"].astype(np.float16), "labels_train": tr["labels"],
                                "logits_train": tr["logits"].astype(np.float16),
                                "feats_known": tk["feats"].astype(np.float16),
                                "feats_unknown": tu["feats"].astype(np.float16)} if args.save_feats else {}),
                            **{f"known_{d}": sk[d] for d in F.DETECTORS},
                            **{f"unknown_{d}": su[d] for d in F.DETECTORS})
    return rows


def main():
    args = parse()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dname = args.dataset_name or Path(args.data).resolve().name
    csv_path = out / "results.csv"
    finished = done_keys(csv_path)
    engine = F.pick_qengine(args.qengine)
    tdev = F.pick_train_device(args.device)
    print(f"[info] torch {torch.__version__} | treino em {tdev} | engine INT8 = {engine}")

    classes, samples = F.discover(args.data, args.layout, args.class_digits)
    print(f"[info] {dname}: {len(classes)} especies, {len(samples)} imagens")
    cache_map = F.build_cache(samples, args.data, args.cache or Path(args.data).resolve().parent / "cache", args.short_side)
    groups = F.load_groups(args.groups, args.data, samples) if args.groups else None

    for seed in args.seeds:
        split = F.make_split(classes, samples, seed, args.known_frac, groups=groups)
        F.save_json(split, out / "splits" / f"{dname}_seed{seed}.json")
        nk, nu = len(split["known_classes"]), len(split["unknown_classes"])
        print(f"\n=== semente {seed}: {nk} conhecidas, {nu} desconhecidas | "
              f"treino {len(split['train'])}, val {len(split['val'])}, "
              f"teste {len(split['test_known'])}+{len(split['test_unknown'])} ===")
        for name in args.backbones:
            vname = lambda v: f"{v}_{args.tag}" if args.tag and v in ("fp32_ft", "qat_int8") else v
            todo = [v for v in args.variants if (dname, seed, name, vname(v)) not in finished]
            if not todo:
                print(f"[skip] {name} semente {seed}: ja avaliado")
                continue
            F.set_seed(seed)
            ckpt = out / "ckpt" / f"{dname}_{name}_seed{seed}_fp32.pt"
            fp32 = F.build_model(name, nk, pretrained=False)
            if ckpt.exists():
                fp32.load_state_dict(torch.load(ckpt, map_location="cpu"))
                print(f"[ckpt] {ckpt.name} carregado")
            else:
                t0 = time.time()
                fp32, val = F.train_fp32(name, nk, split, cache_map, args, tdev)
                ckpt.parent.mkdir(parents=True, exist_ok=True)
                torch.save(fp32.state_dict(), ckpt)
                print(f"[fp32] {name} val_acc={val:.4f} ({(time.time() - t0) / 60:.1f} min)")
            fp32 = fp32.cpu().eval()

            ctx = dict(dataset=dname, seed=seed, backbone=name, n_known=nk, n_unknown=nu,
                       n_test_known=len(split["test_known"]), n_test_unknown=len(split["test_unknown"]),
                       qengine=engine, torch=torch.__version__)
            rng = np.random.RandomState(seed)
            calib = [split["train"][i] for i in
                     rng.choice(len(split["train"]), min(args.calib_images, len(split["train"])), replace=False)]

            for v in todo:
                t0 = time.time()
                if v == "fp32":
                    model = fp32
                elif v == "ptq_int8":
                    model = F.ptq_int8(fp32, calib, cache_map, args, engine)
                elif v == "fp32_ft":
                    F.set_seed(seed)
                    model = F.finetune_fp32(fp32, split, cache_map, args)
                else:
                    F.set_seed(seed)
                    model = F.qat_int8(fp32, split, cache_map, args, engine)
                rows = evaluate(model, name, split, cache_map, args, dict(ctx, variant=vname(v)))
                append_rows(csv_path, rows)
                best = max(rows, key=lambda r: r["auroc"])
                print(f"[{vname(v)}] {name} s{seed}: acc={rows[0]['acc']:.4f} | melhor AUROC={best['auroc']:.4f} "
                      f"({best['detector']}) | {rows[0]['size_mb']:.1f} MB | "
                      f"{rows[0]['latency_ms']:.0f} ms/img | {time.time() - t0:.0f}s")

    print(f"\n[ok] resultados em {csv_path}. Rode: python analyze_exp1.py --results {csv_path}")


if __name__ == "__main__":
    main()
