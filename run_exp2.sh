#!/usr/bin/env bash
# Experimentos novos, em ordem de prioridade. Cada etapa pode ser interrompida e retomada:
# o que ja esta nos CSVs e pulado.
#
# Uso (de dentro de FSR/exp1, com o venv ativado):
#   bash run_exp2.sh            # todas as etapas
#   bash run_exp2.sh 3          # comeca da etapa 3
#   bash run_exp2.sh 2 2        # so a etapa 2
#
# Etapas e tempo estimado no Mac (pela rodada anterior):
#   1. Ablacao da PTQ na macro (MobileNetV3 e ResNet-18, 10 sementes)        ~6 a 8 h
#   2. Base micro: MobileNetV3 e ResNet-18, 10 sementes, 4 variantes         ~17 a 20 h
#   3. MobileNetV2 na macro, 10 sementes, 4 variantes                         ~8 a 10 h
#   4. MobileNetV2 na micro, 10 sementes, 4 variantes                         ~8 a 10 h
#   5. Ablacao da PTQ: MobileNetV2 (macro) e MobileNetV3/V2 (micro)           ~8 a 10 h
#   6. Analises (tabelas, figuras, estatisticas)                              minutos
#
# No Mac, o script se reexecuta sob "caffeinate" para o computador nao dormir.
# Mantenha o notebook na tomada: com a tampa fechada na bateria ele dorme mesmo assim.

set -euo pipefail
cd "$(dirname "$0")"

if [[ "$(uname)" == "Darwin" && -z "${CAFFEINATED:-}" ]]; then
    export CAFFEINATED=1
    exec caffeinate -dims bash "$0" "$@"
fi

FIRST=${1:-1}
LAST=${2:-6}
SEEDS=(0 1 2 3 4 5 6 7 8 9)
MACRO=(--data ../base --dataset-name macro --out results_macro)
# Micro: divisao por grupos (imagens repetidas ou com campos sobrepostos ficam juntas, ver groups/).
# Macro: mantem a divisao por imagem do Experimento 1, para comparar com os resultados ja obtidos.
# O efeito das imagens sobrepostas na macro e medido por check_leakage.py (etapa 6).
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
VARIANTS=(--variants fp32 ptq_int8 fp32_ft qat_int8 --qat-lr 1e-4 --tag lr1e-4 --save-scores --save-feats)
LOG="results_macro/run_exp2_$(date +%Y%m%d_%H%M).log"
mkdir -p results_macro results_micro

run() {  # run <etapa> <descricao> <comando...>
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

# 1. Onde nasce o problema e como corrigir (sem treino: usa os checkpoints FP32 ja salvos)
run 1 "ablacao PTQ, macro, MobileNetV3" \
    python exp2_ptq_ablation.py "${MACRO[@]}" --backbones mobilenet_v3_large --seeds "${SEEDS[@]}" --layer-error
run 1 "ablacao PTQ, macro, ResNet-18 (configuracoes principais)" \
    python exp2_ptq_ablation.py "${MACRO[@]}" --backbones resnet18 --seeds "${SEEDS[@]}" --layer-error \
    --configs base pc w_only_pt w_only_pc fc32

# 2. Segunda base
run 2 "micro, MobileNetV3 e ResNet-18" \
    python run_exp1.py "${MICRO[@]}" --backbones mobilenet_v3_large resnet18 --seeds "${SEEDS[@]}" "${VARIANTS[@]}"

# 3 e 4. Terceira arquitetura (depthwise sem Hardswish nem SE)
run 3 "macro, MobileNetV2" \
    python run_exp1.py "${MACRO[@]}" --backbones mobilenet_v2 --seeds "${SEEDS[@]}" "${VARIANTS[@]}"
run 4 "micro, MobileNetV2" \
    python run_exp1.py "${MICRO[@]}" --backbones mobilenet_v2 --seeds "${SEEDS[@]}" "${VARIANTS[@]}"

# 5. Ablacao nas combinacoes novas
run 5 "ablacao PTQ, macro, MobileNetV2" \
    python exp2_ptq_ablation.py "${MACRO[@]}" --backbones mobilenet_v2 --seeds "${SEEDS[@]}" --layer-error
run 5 "ablacao PTQ, micro, MobileNetV3 e MobileNetV2" \
    python exp2_ptq_ablation.py "${MICRO[@]}" --backbones mobilenet_v3_large mobilenet_v2 --seeds "${SEEDS[@]}" --layer-error
run 5 "ablacao PTQ, micro, ResNet-18 (configuracoes principais)" \
    python exp2_ptq_ablation.py "${MICRO[@]}" --backbones resnet18 --seeds "${SEEDS[@]}" --layer-error \
    --configs base pc w_only_pt w_only_pc fc32

# 6. Analises
for R in results_macro results_micro; do
    if [[ -f "$R/results.csv" ]]; then
        run 6 "analises $R" python analyze_exp1.py --results "$R/results.csv"
        run 6 "estatisticas pareadas $R" python paired_stats.py --results "$R/results.csv"
        run 6 "tabelas e figuras $R" python paper_figures.py --results "$R"
    fi
    if [[ -f "$R/results_ablation.csv" ]]; then
        run 6 "ablacao $R" python analyze_ablation.py --results "$R"
    fi
done
run 6 "imagens sobrepostas: teste de robustez (macro)" python check_leakage.py --results results_macro --groups groups/macro_groups.csv
run 6 "proximas x distantes (macro)" python analyze_near_far.py --results results_macro --taxonomy taxonomy/macro_taxonomy.csv
run 6 "proximas x distantes (micro)" python analyze_near_far.py --results results_micro --taxonomy taxonomy/micro_taxonomy.csv
echo "[ok] log em $LOG"
