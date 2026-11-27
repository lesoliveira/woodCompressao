#!/usr/bin/env bash
# Experimento 6: as camadas depthwise sao a causa da piora? E as outras camadas?
#
# MobileNetV3 e MobileNetV2, duas bases, 10 sementes. Nada e treinado: usa os checkpoints FP32
# e a mesma calibracao da ablacao.
#
# (a) Isolamento das depthwise
#   w_only_dw    so os pesos das depthwise quantizados (por tensor), o resto em FP32
#   w_only_nodw  os pesos de todas as camadas menos as depthwise (por tensor)
#   dw_pt        INT8 completo: por tensor so nas depthwise, por canal no resto
#   dw_pc        INT8 completo: por canal so nas depthwise, por tensor no resto
# (b) Sensibilidade por tipo de camada (so os pesos daquele tipo, por tensor)
#   w_only_stem, w_only_pw, w_only_se (so MobileNetV3), w_only_head
#   w_only_none  referencia: FP32 sem quantizar, avaliado no mesmo dispositivo
#
# As configuracoes "so pesos" sao modelos FP32 e rodam na GPU do Mac (mps), bem mais rapida que a
# CPU para FP32. A etapa 0 confere se a GPU da os mesmos resultados que a CPU. Se nao der, tudo
# roda na CPU (bem mais lento). As INT8 rodam sempre na CPU.
#
# Etapas:
#   0. conferencia da GPU (segundos)
#   1. macro
#   2. micro
#   3. analise: results_dw/dw_isolation.csv
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

FIRST=${1:-0}
LAST=${2:-3}
SEEDS=(0 1 2 3 4 5 6 7 8 9)
CONFIGS=(w_only_none w_only_dw w_only_nodw w_only_stem w_only_pw w_only_se w_only_head dw_pt dw_pc)
MACRO=(--data ../base --dataset-name macro --out results_macro)
MICRO=(--data "../base micro" --dataset-name micro --out results_micro --short-side 768 --groups groups/micro_groups.csv)
LOG="results_macro/run_exp6_$(date +%Y%m%d_%H%M).log"
DEVFILE=results_macro/.exp6_fp32_device

run() {
    local n=$1 desc=$2
    shift 2
    if (( n < FIRST || n > LAST )); then return; fi
    echo -e "\n######## Etapa $n: $desc  ($(date '+%d/%m %H:%M'))" | tee -a "$LOG"
    "$@" 2>&1 | tee -a "$LOG"
}

check_mps() {
python - <<'EOF'
import torch
import fsrood as F
dev = "cpu"
if torch.backends.mps.is_available():
    try:
        torch.manual_seed(0)
        ok = True
        for name in ("mobilenet_v3_large", "mobilenet_v2"):
            m = F.build_model(name, 20, pretrained=False).eval()
            x = torch.randn(12, 3, 224, 224)
            with torch.no_grad():
                a = m(x)
                b = m.to("mps")(x.to("mps")).cpu()
            err = float((a - b).abs().max() / a.abs().max())
            print(f"  {name}: diferenca relativa maxima CPU x GPU = {err:.1e}")
            ok = ok and err < 1e-3
        dev = "mps" if ok else "cpu"
    except Exception as e:
        print("  GPU falhou:", e)
print(f"  modelos FP32 serao avaliados em: {dev}")
open("results_macro/.exp6_fp32_device", "w").write(dev)
EOF
}

run 0 "conferencia da GPU (mps)" check_mps
FP32DEV=$(cat "$DEVFILE" 2>/dev/null || echo cpu)
COMMON=(--backbones mobilenet_v3_large mobilenet_v2 --configs "${CONFIGS[@]}" --seeds "${SEEDS[@]}" --fp32-device "$FP32DEV")

run 1 "macro" python exp2_ptq_ablation.py "${MACRO[@]}" "${COMMON[@]}"
run 2 "micro" python exp2_ptq_ablation.py "${MICRO[@]}" "${COMMON[@]}"
run 3 "analise" python analyze_dw.py
echo -e "\n[ok] fim: $(date '+%d/%m %H:%M'). Log em $LOG" | tee -a "$LOG"
