# Experimento 1: compressão contra detecção de espécies desconhecidas

Pergunta: a quantização INT8 prejudica mais a detecção de espécies desconhecidas do que a acurácia fechada?

Para cada semente e backbone, o script divide as espécies em conhecidas e desconhecidas, treina o modelo FP32 nas conhecidas, gera duas versões INT8 (PTQ e QAT) e avalia as três com cinco detectores pós-hoc. Toda a avaliação roda em CPU com kernels INT8 reais do PyTorch (`qnnpack` em ARM, como Mac M e Raspberry Pi, e `x86` em Intel/AMD). Não há quantização simulada.

## Arquivos

| Arquivo | Conteúdo |
|---|---|
| `fsrood.py` | dados, patches, modelos, treino, PTQ/QAT, detectores e métricas |
| `run_exp1.py` | roda o experimento e grava `results.csv` |
| `analyze_exp1.py` | resumo por configuração, perdas relativas ao FP32, QAT contra controle e gráfico |
| `run_exp2.sh` | experimentos novos em etapas: ablação da PTQ, base micro, MobileNetV2, análises |
| `exp2_ptq_ablation.py` | ablação da PTQ sobre os checkpoints FP32: pesos por canal, observadores, partes em FP32, tamanho da calibração, só pesos, erro bloco a bloco |
| `analyze_ablation.py` | diferenças de cada ablação contra o FP32 e erro de quantização por bloco (`fig_layer_error`) |
| `analyze_near_far.py` | FPR@95 separado para desconhecidas do mesmo gênero de uma conhecida e de gênero diferente, e por grupo (Hardwood/Softwood) |
| `check_leakage.py` | teste de robustez: recalcula as métricas sem as imagens de teste que têm uma imagem sobreposta no treino |
| `groups/` | grupos de imagens repetidas ou sobrepostas (`macro_groups.csv`, `micro_groups.csv`), usados com `--groups` |
| `paper_figures.py` | tabelas LaTeX, figuras e análise de mecanismo para o artigo |
| `run_more_seeds.sh` | sementes 5 a 9 só com as comparações principais (PTQ contra FP32, QAT lr 1e-4 contra controle) |
| `paired_stats.py` | diferenças pareadas por semente, contagem de sinais, teste t e Wilcoxon (`paired_stats.csv`) |
| `run_remaining.sh` | rodadas que faltam (controle da QAT, sementes 3 e 4, QAT com lr 1e-4), com log e sem deixar o Mac dormir |

## Instalação

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Base de dados

A base macroscópica fica em `FSR/base`, ao lado desta pasta `exp1`. São 2.942 imagens JPG de 3264×2448 de 41 espécies, todas numa pasta só. O nome da imagem tem 4 dígitos: os dois primeiros são a classe e os dois últimos o número da imagem (`0101.JPG` é a imagem 01 da classe 01). O script lê a classe do nome (`--layout filename`, detectado automaticamente).

Também funciona com uma pasta por espécie (`--layout folders`), inclusive aninhada. A base microscópica (`FSR/base micro`) tem 112 espécies em `Hardwood/` e `Softwood/`, com 20 imagens PNG de 1024×768 por espécie. Cada pasta com imagens vira uma classe, e o nome da pasta (por exemplo `Softwood/003 Araucaria angustifolia`) traz o gênero usado em `analyze_near_far.py`. Na micro usamos `--short-side 768`, para não reduzir as imagens.

Cada imagem é de uma amostra diferente, então a divisão por imagem não vaza informação entre treino e teste. Para outras bases com várias imagens por amostra, seria preciso adaptar `make_split` para dividir por amostra.

## Como rodar

Teste rápido para ver se tudo funciona (alguns minutos):

```bash
cd FSR/exp1
python run_exp1.py --dataset-name macro --out teste --seeds 0 \
    --backbones mobilenet_v3_large --epochs 2 --qat-epochs 1
```

Experimento completo:

```bash
python run_exp1.py --dataset-name macro --out results_macro \
    --backbones mobilenet_v3_large resnet18 --seeds 0 1 2 3 4 --save-scores
python analyze_exp1.py --results results_macro/results.csv
```

O script pode ser interrompido e retomado. Checkpoints FP32 ficam em `ckpt/`, e combinações já avaliadas no CSV são puladas. Na primeira execução as imagens são redimensionadas e guardadas em `FSR/cache`, que é compartilhado por todas as execuções. Isso leva alguns minutos, porque a base original tem cerca de 8 GB.

O padrão de `--data` é `../base`, então os comandos acima devem ser rodados de dentro de `exp1`.

## O que o experimento faz

- **Divisão:** 50% das espécies conhecidas (`--known-frac`), sorteadas por semente. Nas conhecidas, 60/20/20 de treino, validação e teste por imagem. As desconhecidas aparecem só no teste. As divisões ficam em `splits/`.
- **Patches:** a imagem é redimensionada para lado menor 672 (`--short-side`) e dividida numa grade de patches 224×224 (3×4 em imagens 4:3). O treino usa recortes aleatórios com espelhamento e leve variação de cor. A decisão por imagem usa a média dos logits e das features dos patches.
- **Backbones:** `mobilenet_v3_large`, `resnet18` e, opcionalmente, `mobilenet_v2` (versões quantizáveis do torchvision, pré-treinadas na ImageNet).
- **Variantes:** `fp32`, `ptq_int8` (calibrada com 100 imagens de treino), `qat_int8` (3 épocas de ajuste fino a partir do FP32, em CPU) e `fp32_ft`, o controle da QAT: o mesmo ajuste fino, com as mesmas épocas, lr, otimizador e congelamento de BN, mas sem quantização. A comparação `qat_int8` contra `fp32_ft` separa o efeito da quantização do efeito do ajuste fino extra. Com `--tag`, as variantes `fp32_ft` e `qat_int8` ganham um sufixo no CSV (por exemplo `qat_int8_lr1e-4` com `--qat-lr 1e-4 --tag lr1e-4`).
- **Detectores:** MSP, MaxLogit, energia, kNN na penúltima camada (k = 5) e Mahalanobis com covariância compartilhada (Ledoit-Wolf). kNN e Mahalanobis são ajustados nas features do próprio modelo avaliado.
- **Métricas:** acurácia fechada, AUROC, FPR@95, OSCR, ECE, tamanho do modelo (MB) e latência mediana por imagem com 4 threads (`--threads`).

## Saídas de `analyze_exp1.py`

- `summary.csv`: média e desvio por base, backbone, variante e detector
- `drops.csv` e `drops_summary.csv`: perdas relativas ao FP32, em %. A perda de AUROC é medida em relação à distância ao acaso (AUROC − 0,5). A coluna `gap_pct` é a perda de AUROC menos a perda de acurácia.
- `qat_vs_control.csv`: QAT comparada ao controle `fp32_ft` com o mesmo sufixo (diferenças absolutas de acurácia, AUROC e FPR@95)
- `drops_scatter.png`: perda de acurácia contra perda de AUROC, com a diagonal. Pontos acima da diagonal sustentam a hipótese H1.

## Observações

- O treino FP32 usa GPU ou MPS (Mac) quando disponível (`--device`). PTQ, QAT e toda a avaliação rodam em CPU, porque os kernels INT8 do PyTorch são de CPU.
- A latência medida no Mac serve só de comparação entre variantes. O número que vale para o artigo é o do Raspberry Pi, que pode ser medido depois com os mesmos checkpoints.
- O modo de quantização usado (`torch.ao.quantization`, eager) está marcado como obsoleto nas versões recentes do PyTorch, mas continua funcionando. Para EfficientNet-Lite e outros modelos sem versão quantizável no torchvision, o caminho é a quantização PT2E (`torchao`), que pode entrar num experimento seguinte.

## Imagens repetidas e sobrepostas

Uma comparação por correlação de fase entre todas as imagens de cada espécie encontrou imagens do mesmo campo ou amostra:

- **Micro:** 10 pares de imagens idênticas (mesmo arquivo) e mais 37 pares com campos sobrepostos, quase sempre com números consecutivos. Ao todo, 90 das 2.240 imagens, em 44 grupos.
- **Macro:** nenhuma imagem idêntica, mas 94 pares de fotos da mesma superfície com pequeno deslocamento ou outra exposição. Ao todo, 177 das 2.942 imagens, em 85 grupos.

O limiar foi o maior pico encontrado entre imagens de espécies diferentes. A correlação de fase só detecta campos que se sobrepõem, então fotos da mesma amostra sem sobreposição não aparecem. Exemplos em `groups/exemplos_pares_sobrepostos.jpg`.

Na micro, `run_exp2.sh` usa `--groups groups/micro_groups.csv`: imagens do mesmo grupo ficam sempre na mesma partição. Na macro, a divisão por imagem do Experimento 1 foi mantida, e `check_leakage.py` mostra que tirar do teste as cerca de 11 imagens por semente com uma gêmea no treino não muda as conclusões.
