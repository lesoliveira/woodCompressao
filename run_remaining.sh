#!/usr/bin/env bash
# Rodadas que faltam do Experimento 1, em ordem de prioridade.
# Cada etapa pode ser interrompida e retomada: o que ja esta no results.csv e pulado.
#
# Uso (de dentro de FSR/exp1, com o venv ativado):
#   bash run_remaining.sh            # todas as etapas
#   bash run_remaining.sh 2          # comeca da etapa 2
#
# No Mac, o script se reexecuta sob "caffeinate" para o computador nao dormir.
# Mantenha o notebook na tomada: com a tampa fechada na bateria ele dorme mesmo assim.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
    export CAFFEINATED=1
    exec caffeinate -dims bash "$0" "$@"
fi

START=${1:-1}
OUT=results_macro
LOG="$OUT/run_remaining_$(date +%Y%m%d_%H%M).log"
mkdir -p "$OUT"
COMMON=(--dataset-name macro --out "$OUT" --save-scores)

step() {  # step <numero> <descricao> <args...>
    local n=$1 desc=$2
    shift 2
    if (( n < START )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    python run_exp1.py "${COMMON[@]}" "$@" 2>&1 | tee -a "$LOG"
}

# 1. Controle da QAT nas sementes ja rodadas: mesmo ajuste fino (3 epocas, lr 1e-3), sem quantizacao.
#    E a rodada mais importante: diz se a piora da QAT vem da quantizacao ou do ajuste fino.
step 1 "controle fp32_ft, sementes 0-2" --seeds 0 1 2 --variants fp32_ft

# 2. Sementes 3 e 4, com as variantes originais e o controle.
step 2 "sementes 3-4 completas" --seeds 3 4 --variants fp32 ptq_int8 qat_int8 fp32_ft

# 3. QAT com lr menor (1e-4) e o controle correspondente, nas 5 sementes.
step 3 "QAT e controle com lr 1e-4" --seeds 0 1 2 3 4 --variants fp32_ft qat_int8 \
    --qat-lr 1e-4 --tag lr1e-4

echo -e "\n######## Analise  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
python analyze_exp1.py --results "$OUT/results.csv" 2>&1 | tee -a "$LOG"
echo "[ok] log em $LOG"
