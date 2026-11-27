#!/usr/bin/env bash
# Experimento 6: as camadas depthwise sao a causa da piora?
#
# Quatro configuracoes novas da ablacao, no MobileNetV3 e no MobileNetV2, nas duas bases:
#   w_only_dw    so os pesos das depthwise quantizados (por tensor), o resto em FP32
#   w_only_nodw  os pesos de todas as camadas menos as depthwise (por tensor)
#   dw_pt        INT8 completo: por tensor so nas depthwise, por canal no resto
#   dw_pc        INT8 completo: por canal so nas depthwise, por tensor no resto
# Nao treina nada: usa os checkpoints FP32 e a mesma calibracao da ablacao.
#
# Etapas:
#   1. macro   ~2 a 3 h
#   2. micro   ~2 a 3 h
#   3. analise (minutos): results_dw/dw_isolation.csv
#
# Uso (de dentro da pasta dos scripts, com o venv ativado):
#   bash run_exp6.sh          # tudo
#   bash run_exp6.sh 3 3      # so a analise
# Pode ser interrompido e retomado.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
    export CAFFEINATED=1
    exec caffeinate -dims bash "$0" "$@"
fi
ulimit -n 16384 2>/dev/null || ulimit -n "$(ulimit -Hn)" 2>/dev/null || true
export PYTHONUNBUFFERED=1 PYTHONWARNINGS=ignore

FIRST=${1:-1}
LAST=${2:-3}
SEEDS=(0 1 2 3 4 5 6 7 8 9)
CFG=(--backbones mobilenet_v3_large mobilenet_v2 --configs w_only_dw w_only_nodw dw_pt dw_pc)
MACRO=(--data ../base --dataset-name macro --out results_macro)
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
LOG="results_macro/run_exp6_$(date +%Y%m%d_%H%M).log"

run() {
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

run 1 "macro: isolamento das depthwise" python exp2_ptq_ablation.py "${MACRO[@]}" "${CFG[@]}" --seeds "${SEEDS[@]}"
run 2 "micro: isolamento das depthwise" python exp2_ptq_ablation.py "${MICRO[@]}" "${CFG[@]}" --seeds "${SEEDS[@]}"
run 3 "analise" python analyze_dw.py
echo -e "\n[ok] fim: $(date '+%d/%m %H:%M'). Log em $LOG" | tee -a "$LOG"
