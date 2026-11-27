"""Varredura do ponto de operacao: FPR em varios niveis de TPR e AUROC, a partir dos escores
por imagem gravados com --save-scores (results_*/scores/*.npz).

Uso:
    python tpr_sweep.py                       # results_macro e results_micro -> results_tpr_sweep/
    python tpr_sweep.py --results results_macro --out results_tpr_sweep
"""
import argparse
import csv
import glob
import os
import re
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

TPRS = [0.80, 0.85, 0.90, 0.95, 0.98, 0.99]
NETS = "mobilenet_v3_large|mobilenet_v2|resnet18|efficientnet_lite0"
VARIANTS = ("fp32", "ptq_int8", "ptq_int8_pc")
DETECTORS = ("msp", "maxlogit", "energy", "knn", "mahalanobis")


def auroc(k, u):
    r = rankdata(np.r_[k, u])
    return float((r[:len(k)].sum() - len(k) * (len(k) + 1) / 2) / (len(k) * len(u)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", default=["results_macro", "results_micro"])
    ap.add_argument("--out", default="results_tpr_sweep")
    args = ap.parse_args()

    rows = []
    for res in args.results:
        for f in sorted(glob.glob(os.path.join(res, "scores", "*.npz"))):
            m = re.match(rf"(\w+?)_s(\d+)_({NETS})_(.+)\.npz$", os.path.basename(f))
            if not m or m.group(4) not in VARIANTS:
                continue
            z = np.load(f)
            for d in DETECTORS:
                if f"known_{d}" not in z:
                    continue
                k, u = z[f"known_{d}"], z[f"unknown_{d}"]
                row = dict(dataset=m.group(1), seed=int(m.group(2)), backbone=m.group(3),
                           variant=m.group(4), detector=d, auroc=auroc(k, u))
                for t in TPRS:
                    # mesmo criterio de fsrood.fpr_at_tpr
                    row[f"fpr{int(round(t * 100))}"] = float((u >= np.quantile(k, 1 - t)).mean())
                rows.append(row)
    if not rows:
        raise SystemExit("nenhum escore encontrado (rode com --save-scores)")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "tpr_sweep.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"[ok] {len(rows)} linhas em {out / 'tpr_sweep.csv'}")


if __name__ == "__main__":
    main()
