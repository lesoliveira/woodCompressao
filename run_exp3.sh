#!/usr/bin/env bash
# Experimento 3: QAT com pesos POR CANAL (MobileNetV3 e MobileNetV2, macro e micro, 10 sementes).
#
# Reaproveita os checkpoints FP32 e o controle sem quantizacao (fp32_ft_lr1e-4) ja calculados.
# A unica variante nova e qat_int8_pc_lr1e-4: o mesmo ajuste fino da QAT anterior (3 epocas,
# lr 1e-4, mesma ordem dos dados), mudando so o fake-quant dos pesos para por canal.
# Comparacao principal: qat_int8_pc_lr1e-4 contra fp32_ft_lr1e-4.
#
# Uso (de dentro da pasta dos scripts, com o venv ativado):
#   bash run_exp3.sh          # tudo (~12 a 15 h no Mac)
#   bash run_exp3.sh 2 2      # so a etapa 2
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
LAST=${2:-3}
SEEDS=(0 1 2 3 4 5 6 7 8 9)
MACRO=(--data ../base --dataset-name macro --out results_macro)
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
QATPC=(--variants qat_int8 --qat-lr 1e-4 --qat-per-channel --tag pc_lr1e-4 --save-scores --save-feats)
LOG="results_macro/run_exp3_$(date +%Y%m%d_%H%M).log"
mkdir -p results_macro results_micro

run() {
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

run 1 "QAT por canal, macro" \
    python run_exp1.py "${MACRO[@]}" --backbones mobilenet_v3_large mobilenet_v2 --seeds "${SEEDS[@]}" "${QATPC[@]}"
run 2 "QAT por canal, micro" \
    python run_exp1.py "${MICRO[@]}" --backbones mobilenet_v3_large mobilenet_v2 --seeds "${SEEDS[@]}" "${QATPC[@]}"

for R in results_macro results_micro; do
    run 3 "estatisticas pareadas, QAT por canal ($R)" \
        python paired_stats.py --results "$R/results.csv" --out-name paired_stats_qat_pc.csv \
        --pairs qat_int8_pc_lr1e-4:fp32_ft_lr1e-4 qat_int8_lr1e-4:fp32_ft_lr1e-4 qat_int8_pc_lr1e-4:fp32
done
echo "[ok] log em $LOG"
