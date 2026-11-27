#!/usr/bin/env bash
# Experimento 5: escores por imagem da PTQ por canal das redes do torchvision, para a varredura
# do ponto de operacao (FPR em TPR de 80% a 99%).
#
# Nao treina nada: carrega os checkpoints FP32 ja salvos (results_*/ckpt), quantiza com pesos por
# canal (mesma calibracao de 100 imagens da ablacao) e grava os escores. A EfficientNet-Lite0 ja
# tem esses escores (Experimento 4).
#
# Etapas:
#   1. PTQ por canal, MobileNetV3, MobileNetV2 e ResNet-18, macro e micro, 10 sementes  ~4 h
#   2. Varredura de TPR e conferencia com a ablacao                                  minutos
#
# Uso (de dentro da pasta dos scripts, com o venv ativado):
#   bash run_exp5.sh          # etapas 1 e 2
#   bash run_exp5.sh 2 2      # so a etapa 2
#
# Pode ser interrompido e retomado: o que ja esta no results.csv e pulado.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
    export CAFFEINATED=1
    exec caffeinate -dims bash "$0" "$@"
fi
ulimit -n 16384 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true

FIRST=${1:-1}
LAST=${2:-2}
SEEDS=(0 1 2 3 4 5 6 7 8 9)
BB=(--backbones mobilenet_v3_large mobilenet_v2 resnet18)
MACRO=(--data ../base --dataset-name macro --out results_macro)
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
LOG="results_macro/run_exp5_$(date +%Y%m%d_%H%M).log"
mkdir -p results_macro results_micro

for d in results_macro results_micro; do
    n=$(ls "$d"/ckpt/*_fp32.pt 2>/dev/null | grep -vc efficientnet || true)
    if (( n < 30 )); then echo "Faltam checkpoints FP32 em $d/ckpt ($n de 30)."; exit 1; fi
done

run() {
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

run 1 "macro: PTQ por canal com escores" \
    python run_exp1.py "${MACRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" --variants ptq_int8_pc --save-scores
run 1 "micro: PTQ por canal com escores" \
    python run_exp1.py "${MICRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" --variants ptq_int8_pc --save-scores

run 2 "varredura de TPR" python tpr_sweep.py
run 2 "conferencia: PTQ por canal (Exp. 5) contra abl_pc (ablacao)" python - <<'EOF'
import pandas as pd
for ds in ("macro", "micro"):
    r = pd.read_csv(f"results_{ds}/results.csv")
    a = pd.read_csv(f"results_{ds}/results_ablation.csv")
    r = r[(r.variant == "ptq_int8_pc") & (r.backbone != "efficientnet_lite0")]
    a = a[a.variant == "abl_pc"]
    k = ["seed", "backbone", "detector"]
    m = r.merge(a, on=k, suffixes=("", "_abl"))
    for c in ("acc", "auroc", "fpr95"):
        m[f"d_{c}"] = (m[c] - m[f"{c}_abl"]).abs()
    print(ds, len(m), "pares | diferenca maxima:",
          {c: round(float(m[f"d_{c}"].max()), 4) for c in ("acc", "auroc", "fpr95")})
EOF
echo -e "\n[ok] fim: $(date '+%d/%m %H:%M'). Log em $LOG" | tee -a "$LOG"
