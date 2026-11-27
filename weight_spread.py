#!/usr/bin/env python3
"""
Por que algumas arquiteturas sofrem tanto com pesos quantizados por tensor?

Mede, nos checkpoints FP32 (com BN ja fundida nas convolucoes, como na quantizacao real),
o quanto a escala dos pesos varia entre os canais de saida de cada camada. Com quantizacao
por tensor, todos os canais dividem uma unica escala, dada pelo canal de maior amplitude:
canais de pesos pequenos ficam com poucos niveis, ou todos zerados.

Por camada (conv e linear):
  spread          max_c(amax_c) / mediana_c(amax_c), amax_c = max |w| do canal c
  err_pt, err_pc  erro relativo ||Q(W) - W|| / ||W|| com INT8 simetrico por tensor e por canal
  dead_frac       fracao de canais com todos os pesos arredondados para zero (por tensor)
  depthwise       camada depthwise (groups == canais de saida > 1)

Resumo por modelo: medias sobre as camadas (todas e so as depthwise). Com --stats, cruza o
resumo com a piora do FPR@95 e da acuracia na PTQ por tensor (paired_stats*.csv).

Uso:
  python weight_spread.py --ckpts results_macro/ckpt/*.pt results_micro/ckpt/*.pt \\
      --stats results_macro/paired_stats.csv results_macro/paired_stats_effnet.csv \\
              results_micro/paired_stats.csv results_micro/paired_stats_effnet.csv \\
      --out weight_spread
"""

import argparse
import copy
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

import fsrood as F


def fused_model(name, n_classes, ckpt):
    m = F.build_model(name, n_classes, pretrained=False)
    m.load_state_dict(torch.load(ckpt, map_location="cpu"))
    m = copy.deepcopy(m).eval()
    if name in F.FX_MODELS:
        from torch.ao.quantization.quantize_fx import fuse_fx
        return fuse_fx(m)
    m.fuse_model(is_qat=False)
    return m


def q_sym(w, scale):
    return torch.clamp(torch.round(w / scale), -128, 127) * scale


@torch.no_grad()
def layer_stats(model):
    rows = []
    for lname, mod in model.named_modules():
        if not isinstance(mod, (nn.Conv2d, nn.Linear)):
            continue
        w = mod.weight.detach().float()
        amax = w.abs().flatten(1).amax(1).clamp(min=1e-12)
        s_pt = amax.max() / 127
        s_pc = (amax / 127).view(-1, *([1] * (w.dim() - 1)))
        wq_pt, wq_pc = q_sym(w, s_pt), q_sym(w, s_pc)
        dead = (wq_pt.flatten(1).abs().amax(1) == 0).float().mean()
        rows.append(dict(layer=lname, kind=type(mod).__name__, out_ch=w.shape[0],
                         depthwise=bool(isinstance(mod, nn.Conv2d) and mod.groups > 1 and mod.groups == w.shape[0]),
                         spread=float(amax.max() / amax.median()),
                         err_pt=float((wq_pt - w).norm() / w.norm()),
                         err_pc=float((wq_pc - w).norm() / w.norm()),
                         dead_frac=float(dead)))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--stats", nargs="*", default=[], help="paired_stats*.csv para cruzar com a piora")
    ap.add_argument("--out", default="weight_spread")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for ck in sorted(args.ckpts):
        stem = Path(ck).stem
        mt = re.match(r"(?P<ds>[a-z]+)_(?P<bb>.+)_seed(?P<seed>\d+)_fp32$", stem)
        if not mt:
            continue
        ds, bb, seed = mt["ds"], mt["bb"], int(mt["seed"])
        sd = torch.load(ck, map_location="cpu")
        n_classes = [v for k, v in sd.items() if k.endswith("weight") and v.dim() == 2][-1].shape[0]
        m = fused_model(bb, n_classes, ck)
        for r in layer_stats(m):
            rows.append(dict(dataset=ds, backbone=bb, seed=seed, **r))
        print(f"[ok] {stem}")
    lay = pd.DataFrame(rows)
    lay.to_csv(out / "weight_spread_layers.csv", index=False)

    def summ(g):
        dw = g[g.depthwise]
        return pd.Series(dict(
            n_layers=len(g), n_depthwise=len(dw),
            spread_median=g.spread.median(), spread_dw_median=dw.spread.median() if len(dw) else np.nan,
            err_pt_mean=g.err_pt.mean(), err_pc_mean=g.err_pc.mean(),
            err_pt_dw_mean=dw.err_pt.mean() if len(dw) else np.nan,
            err_pt_max=g.err_pt.max(), dead_frac_mean=g.dead_frac.mean(),
            dead_frac_dw_mean=dw.dead_frac.mean() if len(dw) else np.nan))
    sm = lay.groupby(["dataset", "backbone", "seed"]).apply(summ, include_groups=False).reset_index()
    model = sm.groupby(["dataset", "backbone"]).mean(numeric_only=True).drop(columns="seed").reset_index()

    if args.stats:
        st = pd.concat([pd.read_csv(f) for f in args.stats])
        st = st[(st.A == "ptq_int8") & (st.B == "fp32")].drop_duplicates(["dataset", "backbone", "detector"])
        piv = st.pivot_table(index=["dataset", "backbone"], columns="detector", values="fpr95_diff_pp")
        piv.columns = [f"d_fpr95_{c}" for c in piv.columns]
        acc = st[st.detector == "msp"].set_index(["dataset", "backbone"]).acc_diff_pp.rename("d_acc")
        model = model.merge(piv.join(acc).reset_index(), on=["dataset", "backbone"], how="left")
        from scipy import stats as S
        print("\nCorrelacao de Spearman entre a medida de pesos e a piora com PTQ por tensor "
              f"({len(model)} combinacoes base x arquitetura):")
        for x in ["spread_dw_median", "spread_median", "err_pt_mean", "err_pt_dw_mean", "dead_frac_mean"]:
            ok = model[[x, "d_fpr95_energy", "d_acc"]].dropna() if x in model else []
            if len(ok) >= 4:
                r1 = S.spearmanr(ok[x], ok.d_fpr95_energy)[0]
                r2 = S.spearmanr(ok[x], ok.d_acc)[0]
                print(f"  {x:18s}  com dFPR95(energia): {r1:+.2f}   com dAcc: {r2:+.2f}")
    model.round(4).to_csv(out / "weight_spread_models.csv", index=False)
    pd.set_option("display.width", 250)
    print(model.round(3).to_string(index=False))
    print(f"\n[ok] {out}")


if __name__ == "__main__":
    main()
