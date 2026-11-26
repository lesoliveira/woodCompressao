#!/usr/bin/env bash
# Experimento 4: quarta arquitetura, EfficientNet-Lite0 (timm), macro e micro, 10 sementes.
#
# A EfficientNet-Lite0 nao tem versao quantizavel no torchvision, entao e quantizada pelo modo FX
# do PyTorch, com os mesmos qconfigs (qnnpack) e kernels INT8 reais das outras redes. Como na
# MobileNetV2 quantizavel do torchvision, o ReLU6 e trocado por ReLU antes do ajuste fino.
#
# Etapas:
#   1. FP32, PTQ por tensor (padrao) e PTQ por canal            ~6 a 8 h   (essencial)
#   2. Controle do ajuste fino, QAT por tensor e QAT por canal   ~15 a 20 h (opcional)
#   3. Estatisticas pareadas                                     minutos
#
# Uso (de dentro da pasta dos scripts, com o venv ativado):
#   pip install timm          # uma vez; os pesos ImageNet sao baixados do Hugging Face no 1o uso
#   bash run_exp4.sh          # etapas 1 a 3
#   bash run_exp4.sh 1 1      # so a etapa 1
#   bash run_exp4.sh 3 3      # so as estatisticas (por exemplo, depois de rodar so a etapa 1)
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
BB=(--backbones efficientnet_lite0)
MACRO=(--data ../base --dataset-name macro --out results_macro)
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
SAVE=(--save-scores --save-feats)
LOG="results_macro/run_exp4_$(date +%Y%m%d_%H%M).log"
mkdir -p results_macro results_micro

python -c "import timm" 2>/dev/null || { echo "Falta o timm: rode 'pip install timm' no venv."; exit 1; }

run() {
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

# (sem "declare -n": o bash do macOS e a versao 3.2)
run 1 "EfficientNet-Lite0, macro: FP32 e PTQ por tensor e por canal" \
    python run_exp1.py "${MACRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" --variants fp32 ptq_int8 ptq_int8_pc "${SAVE[@]}"
run 1 "EfficientNet-Lite0, micro: FP32 e PTQ por tensor e por canal" \
    python run_exp1.py "${MICRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" --variants fp32 ptq_int8 ptq_int8_pc "${SAVE[@]}"

QATT=(--variants fp32_ft qat_int8 --qat-lr 1e-4 --tag lr1e-4)
QATC=(--variants qat_int8 --qat-lr 1e-4 --qat-per-channel --tag pc_lr1e-4)
run 2 "EfficientNet-Lite0, macro: controle e QAT por tensor" \
    python run_exp1.py "${MACRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" "${QATT[@]}" "${SAVE[@]}"
run 2 "EfficientNet-Lite0, micro: controle e QAT por tensor" \
    python run_exp1.py "${MICRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" "${QATT[@]}" "${SAVE[@]}"
run 2 "EfficientNet-Lite0, macro: QAT por canal" \
    python run_exp1.py "${MACRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" "${QATC[@]}" "${SAVE[@]}"
run 2 "EfficientNet-Lite0, micro: QAT por canal" \
    python run_exp1.py "${MICRO[@]}" "${BB[@]}" --seeds "${SEEDS[@]}" "${QATC[@]}" "${SAVE[@]}"

for R in results_macro results_micro; do
    run 3 "estatisticas pareadas, EfficientNet-Lite0 ($R)" \
        python paired_stats.py --results "$R/results.csv" --out-name paired_stats_effnet.csv \
        --pairs ptq_int8:fp32 ptq_int8_pc:fp32 qat_int8_lr1e-4:fp32_ft_lr1e-4 qat_int8_pc_lr1e-4:fp32_ft_lr1e-4
done
echo "[ok] log em $LOG"
