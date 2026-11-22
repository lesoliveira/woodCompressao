#!/usr/bin/env python3
"""
Desconhecidas proximas e distantes: a quantizacao prejudica mais o caso dificil?

Para cada imagem desconhecida do teste, separa as desconhecidas em subconjuntos:
  near_genus / far_genus     o genero da especie desconhecida aparece (ou nao) entre as conhecidas
  near_family / far_family   idem para a familia (precisa de --taxonomy)
  group_<nome>               por grupo da desconhecida (pasta de cima, ex.: Hardwood/Softwood)

O FPR@95 e calculado com o limiar de 95% de TPR das conhecidas (o mesmo de sempre) e
separado por subconjunto de desconhecidas. Depois compara as variantes pareadas:
  ptq_int8 contra fp32   e   qat_int8_lr1e-4 contra fp32_ft_lr1e-4

Taxonomia: --taxonomy com um CSV de colunas id,family,species (como taxonomy/macro_taxonomy.csv
e taxonomy/micro_taxonomy.csv, tiradas dos artigos MVA 2014 e MVA 2012). O id e o numero no
inicio do nome da classe ("01" na macro, "003 Araucaria angustifolia" na micro). Sem --taxonomy,
o genero sai do proprio nome da pasta e a familia fica de fora.

Saidas: near_far.csv (por semente) e near_far_summary.csv na pasta de resultados.

Uso:
  python analyze_near_far.py --results results_micro --taxonomy taxonomy/micro_taxonomy.csv
  python analyze_near_far.py --results results_macro --taxonomy taxonomy/macro_taxonomy.csv
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

DETS = ["msp", "maxlogit", "energy", "knn", "mahalanobis"]
PAIRS = [("ptq_int8", "fp32"), ("qat_int8_lr1e-4", "fp32_ft_lr1e-4")]


def class_id(cls):
    m = re.match(r"\s*(\d+)", cls.split("/")[-1])
    return int(m.group(1)) if m else None


def species_name(cls, tax):
    if tax is not None and class_id(cls) in tax:
        return tax[class_id(cls)]["species"]
    return re.sub(r"^\d+\s*", "", cls.split("/")[-1])     # "003 Araucaria angustifolia" -> nome


def genus(cls, tax):
    sp = species_name(cls, tax).strip()
    return sp.split()[0].lower() if sp and not sp.isdigit() else None


def family(cls, tax):
    if tax is None or class_id(cls) not in tax:
        return None
    return str(tax[class_id(cls)]["family"]).lower()


def group(cls):
    return cls.split("/")[0] if "/" in cls else "all"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results", required=True)
    ap.add_argument("--taxonomy", default=None, help="CSV com colunas id,family,species")
    args = ap.parse_args()
    res = Path(args.results)
    names = None                                   # taxonomia: id -> {family, species}
    if args.taxonomy:
        t = pd.read_csv(args.taxonomy)
        names = {int(r.id): {"family": r.family, "species": r.species} for r in t.itertuples()}

    rows = []
    for sp in sorted((res / "splits").glob("*_seed*.json")):
        split = json.load(open(sp))
        dname, seed = sp.stem.rsplit("_seed", 1)
        seed = int(seed)
        known_genera = {genus(c, names) for c in split["known_classes"]} - {None}
        known_families = {family(c, names) for c in split["known_classes"]} - {None}
        # classe de cada imagem desconhecida: test_unknown guarda o caminho; a classe e a pasta ou o prefixo
        unk_cls = []
        for p, _ in split["test_unknown"]:
            pp = Path(p)
            cls = next((c for c in split["unknown_classes"] if str(pp.parent).endswith("/" + c) or str(pp.parent) == c), None)
            if cls is None:                                   # layout por nome de arquivo
                cls = next((c for c in split["unknown_classes"] if pp.stem.startswith(c)), "?")
            unk_cls.append(cls)
        g_unk = np.array([genus(c, names) for c in unk_cls], dtype=object)
        near = np.array([g in known_genera for g in g_unk])
        f_unk = [family(c, names) for c in unk_cls]
        near_f = np.array([f in known_families for f in f_unk])
        grp = np.array([group(c) for c in unk_cls])
        if names is None and all(g is None for g in g_unk):
            print(f"[aviso] {sp.name}: classes sem nome de genero. Use --taxonomy para essa base.")
        subsets = {"all": np.ones(len(near), bool), "near_genus": near, "far_genus": ~near}
        if any(f is not None for f in f_unk):
            subsets.update({"near_family": near_f, "far_family": ~near_f})
        for gname in sorted(set(grp)):
            if gname != "all":
                subsets[f"group_{gname}"] = grp == gname
        for f in sorted((res / "scores").glob(f"{dname}_s{seed}_*.npz")):
            rest = f.stem[len(f"{dname}_s{seed}_"):]
            bb = next((b for b in ("mobilenet_v3_large", "mobilenet_v2", "resnet18") if rest.startswith(b)), None)
            if bb is None:
                continue
            variant = rest[len(bb) + 1:]
            z = np.load(f)
            for d in DETS:
                k, u = z["known_" + d], z["unknown_" + d]
                if len(u) != len(near):
                    raise RuntimeError(f"{f.name}: {len(u)} desconhecidas no npz, {len(near)} na divisao")
                thr = np.quantile(k, 0.05)
                for sname, m in subsets.items():
                    if m.sum() == 0:
                        continue
                    rows.append(dict(dataset=dname, seed=seed, backbone=bb, variant=variant, detector=d,
                                     subset=sname, n_unknown_images=int(m.sum()),
                                     n_unknown_species=len({c for c, mm in zip(unk_cls, m) if mm}),
                                     fpr95=float((u[m] >= thr).mean())))
    df = pd.DataFrame(rows)
    df.to_csv(res / "near_far.csv", index=False)

    out = []
    for (bb, d, sname), g in df.groupby(["backbone", "detector", "subset"]):
        for q, r in PAIRS:
            A = g[g.variant == q].set_index("seed").fpr95
            B = g[g.variant == r].set_index("seed").fpr95
            s = sorted(set(A.index) & set(B.index))
            if not s:
                continue
            dd = (A.loc[s] - B.loc[s]).to_numpy() * 100
            out.append(dict(backbone=bb, comparison=f"{q} vs {r}", detector=d, subset=sname, n_seeds=len(s),
                            n_species=g[g.variant == r].n_unknown_species.mean(),
                            fpr95_ref=B.loc[s].mean() * 100, fpr95_int8=A.loc[s].mean() * 100,
                            d_fpr95_pp=dd.mean(), seeds_worse=int((dd > 0).sum()),
                            p_wilcoxon=1.0 if np.allclose(dd, 0) else stats.wilcoxon(dd, zero_method="zsplit").pvalue))
    summ = pd.DataFrame(out)
    summ.round(4).to_csv(res / "near_far_summary.csv", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(summ.round(2).to_string(index=False))
    print(f"\n[ok] {res / 'near_far_summary.csv'}")


if __name__ == "__main__":
    main()
