#!/usr/bin/env bash
# Sementes extras (5 a 9) do Experimento 1, so com as duas comparacoes principais:
#   PTQ INT8 contra FP32
#   QAT INT8 (lr 1e-4) contra o controle fp32_ft (mesmo ajuste fino, sem quantizacao)
# As variantes com lr 1e-3 ficam de fora.
#
# Uso (de dentro de FSR/exp1, com o venv ativado):
#   bash run_more_seeds.sh              # sementes 5 a 9
#   bash run_more_seeds.sh 7 8 9        # so as sementes indicadas
#
# Pode ser interrompido e retomado: o que ja esta no results.csv e pulado.
# No Mac, o script se reexecuta sob "caffeinate" para o computador nao dormir.
# Mantenha o notebook na tomada: com a tampa fechada na bateria ele dorme mesmo assim.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
    export CAFFEINATED=1
    exec caffeinate -dims bash "$0" "$@"
fi

SEEDS=("$@")
if (( ${#SEEDS[@]} == 0 )); then SEEDS=(5 6 7 8 9); fi
OUT=results_macro
LOG="$OUT/run_more_seeds_$(date +%Y%m%d_%H%M).log"
mkdir -p "$OUT"

for s in "${SEEDS[@]}"; do
    echo -e "\n######## Semente $s  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    python run_exp1.py --dataset-name macro --out "$OUT" --save-scores --seeds "$s" \
        --variants fp32 ptq_int8 fp32_ft qat_int8 --qat-lr 1e-4 --tag lr1e-4 2>&1 | tee -a "$LOG"
done

echo -e "\n######## Analise  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
python analyze_exp1.py --results "$OUT/results.csv" 2>&1 | tee -a "$LOG"
python paired_stats.py --results "$OUT/results.csv" 2>&1 | tee -a "$LOG"
echo "[ok] log em $LOG"
