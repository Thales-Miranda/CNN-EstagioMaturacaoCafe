# Classificação de Maturação de Frutos de Café

> Universidade Federal de Uberlândia — UFU  
> Disciplina: Mineração de Dados — FACOM  
> Prof. Murillo Guimarães Carneiro  
> Aluno: Thales Elias Miranda — 12221GIN016

---

## Sobre o Projeto

A ideia deste projeto nasceu de um problema real da cafeicultura brasileira: como saber o momento certo de colher os frutos? Hoje, essa decisão depende da experiência visual do agricultor — um processo subjetivo, lento e que pode levar a perdas significativas de qualidade e valor.

A proposta foi desenvolver um sistema capaz de classificar automaticamente o estágio de maturação de frutos de café a partir de fotos tiradas pelo próprio agricultor, usando redes neurais treinadas em Python com PyTorch.

O projeto foi construído de forma gradual, evoluindo de uma arquitetura simples (MLP) até uma abordagem moderna com Transfer Learning (EfficientNet), passando por uma CNN desenvolvida do zero.

---

## As 5 Classes de Maturação

| Índice | Classe | Descrição |
|--------|------------|--------------------------------------------------|
| 0 | Verde | Fruto imaturo, coloração verde |
| 1 | Verde cana | Fruto em transição, início de amadurecimento |
| 2 | Cereja | Fruto maduro, coloração vermelha intensa |
| 3 | Passa | Fruto em processo de secagem na planta |
| 4 | Seco | Fruto completamente seco na planta |

---

## Estrutura do Projeto

```
Mineração Café/
├── data/
│   ├── train/
│   │   ├── Cereja/
│   │   ├── Passa/
│   │   ├── Seco/
│   │   ├── Verde/
│   │   └── Verde cana/
│   └── test/
│       └── *.jpg        (15 imagens sem rótulo — submissão Kaggle)
├── resultados/
│   ├── mlp_melhor_modelo.pth
│   ├── cnn_melhor_modelo.pth
│   ├── efficientnet_melhor_modelo.pth
│   ├── mlp_metricas.json
│   ├── cnn_metricas.json
│   ├── efficientnet_metricas.json
│   └── submissao.csv
├── dados.py
├── mlp.py
├── cnn.py
├── efficientnet.py
├── treino.py
├── treino_mlp.py
├── treino_cnn.py
├── treino_efficientnet.py
├── submissao.py
└── README.md
```

---

## O que cada arquivo faz

| Arquivo | Responsabilidade |
|---|---|
| `dados.py` | Lê as imagens do disco, aplica Data Augmentation e entrega os dados em batches para a rede |
| `mlp.py` | Define a arquitetura da MLP — rede baseline que trata a imagem como vetor de pixels |
| `cnn.py` | Define a CNN com 4 blocos convolucionais — detecta padrões visuais como bordas e texturas |
| `efficientnet.py` | Carrega a EfficientNet-B0 pré-treinada e adapta para as 5 classes de café |
| `treino.py` | Motor de treinamento compartilhado entre todos os modelos — inclui Early Stopping e métricas |
| `treino_mlp.py` | Script para treinar a MLP |
| `treino_cnn.py` | Script para treinar a CNN |
| `treino_efficientnet.py` | Script para treinar a EfficientNet com Transfer Learning em 2 estágios |
| `submissao.py` | Gera o CSV de predições para submissão no Kaggle (usa EfficientNet por padrão) |

---

## Instalação

### Requisitos

- Python 3.10 ou superior
- Windows, Linux (Fedora 30–36) ou macOS
- Mínimo 4 GB de RAM
- GPU NVIDIA (opcional — acelera bastante o treino)

### Passo a passo

```bash
# 1. Criar ambiente virtual
python -m venv venv

# 2. Ativar o ambiente
# Windows:
venv\Scripts\activate
# Linux / macOS:
source venv/bin/activate

# 3. Instalar as dependências
pip install torch torchvision scikit-learn pillow
```

---

## Como Executar

### 1. Treinar a MLP (baseline obrigatório)

```bash
python treino_mlp.py
```

### 2. Treinar a CNN

```bash
python treino_cnn.py
```

### 3. Treinar a EfficientNet (Transfer Learning)

```bash
python treino_efficientnet.py
```

### 4. Gerar o CSV de submissão

```bash
# EfficientNet — melhor resultado (padrão)
python submissao.py

# CNN
python submissao.py --modelo cnn

# MLP
python submissao.py --modelo mlp
```

O arquivo `resultados/submissao.csv` estará pronto para enviar no Kaggle.

---

## As Arquiteturas

### MLP — O Ponto de Partida

A MLP foi o primeiro modelo implementado, como exigido pelo projeto. Ela achata a imagem inteira em um vetor de pixels e passa por camadas de neurônios conectados. É simples e serve bem como referência, mas tem uma limitação fundamental: ao achatar a imagem, ela perde toda a informação espacial — não sabe que pixels vizinhos formam padrões, bordas ou cores.

| Detalhe | Valor |
|---|---|
| Entrada | 3 × 64 × 64 = 12.288 pixels |
| Camadas ocultas | 512 → 256 → 128 neurônios |
| Regularização | BatchNorm + ReLU + Dropout(0.5) |
| Parâmetros | 6.458.629 |
| Otimizador | Adam (lr=0.001, weight_decay=0.001) |
| Scheduler | ReduceLROnPlateau |
| Early Stopping | Paciência de 30 épocas |

---

### CNN — Aprendendo a Enxergar

A CNN foi o segundo modelo, desenvolvida do zero com 4 blocos convolucionais. Diferente da MLP, ela preserva a estrutura espacial da imagem e aplica filtros que deslizam sobre ela detectando bordas, texturas e cores. Cada bloco aprende padrões progressivamente mais complexos — do contorno do fruto até sua cor e textura superficial.

| Detalhe | Valor |
|---|---|
| Entrada | 3 × 128 × 128 pixels |
| Blocos convolucionais | 4 blocos: 32 → 64 → 128 → 256 filtros |
| Cada bloco | Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool |
| Classificador | 4096 → 512 → 128 → 5 classes |
| Parâmetros | 3.339.429 |
| Otimizador | Adam (lr=0.001, weight_decay=0.0001) |
| Early Stopping | Paciência de 30 épocas |

---

### EfficientNet — Transfer Learning

O terceiro e melhor modelo. Em vez de treinar do zero, aproveitamos a EfficientNet-B0 já treinada em 1,2 milhão de imagens do ImageNet. Ela já sabe reconhecer bordas, texturas, formas e cores — só precisou aprender a aplicar esse conhecimento para distinguir os frutos de café.

O treinamento foi feito em 2 estágios:

**Estágio 1 — Backbone congelado (10 épocas)**
O backbone fica congelado para não destruir o conhecimento pré-treinado. Apenas o classificador final é treinado com LR alto (0.001).

**Estágio 2 — Fine-tuning completo (40 épocas)**
Toda a rede é descongelada e treinada com LR baixo (0.0001) para especializar o conhecimento no domínio de café.

| Detalhe | Valor |
|---|---|
| Base | EfficientNet-B0 (ImageNet) |
| Entrada | 3 × 224 × 224 pixels |
| Classificador | 1280 → 5 classes |
| Dropout | 0.3 |
| LR Estágio 1 | 0.001 |
| LR Estágio 2 | 0.0001 |
| Scheduler E2 | CosineAnnealingLR |

---

## Pré-processamento e Data Augmentation

Com apenas 35 imagens de treino, o Data Augmentation foi essencial para evitar que os modelos simplesmente decorassem os exemplos. A cada época, cada imagem é transformada de forma diferente:

| Transformação | O que simula |
|---|---|
| Resize + RandomCrop | Diferentes enquadramentos da foto |
| RandomHorizontalFlip | Planta fotografada de lados diferentes |
| RandomVerticalFlip | Ângulos variados de captura |
| RandomRotation(30°) | Agricultor girando o celular |
| ColorJitter | Variações de luz, sombra e exposição |
| Normalize (ImageNet) | Padronização dos valores para o treino |

---

## Resultados

Avaliação no conjunto de validação (7 imagens — 20% do treino):

| Modelo | Precisão | Recall | F1-macro | Melhora |
|---|---|---|---|---|
| MLP (baseline) | 0.20 | 0.40 | 0.27 | — |
| CNN (do zero) | 0.63 | 0.70 | 0.63 | +133% |
| EfficientNet (Transfer Learning) | 0.93 | 0.90 | **0.89** | **+230%** |

### Detalhamento por classe — EfficientNet

| Classe | Precisão | Recall | F1 |
|---|---|---|---|
| Verde | 1.00 | 1.00 | 1.00 |
| Verde cana | 1.00 | 1.00 | 1.00 |
| Cereja | 1.00 | 1.00 | 1.00 |
| Passa | 0.67 | 1.00 | 0.80 |
| Seco | 1.00 | 0.50 | 0.67 |

---

## Métricas de Avaliação

- **Precisão (macro):** de tudo que o modelo classificou como Cereja, quantos eram realmente Cereja?
- **Recall (macro):** de todas as Cerejas reais, quantas o modelo identificou corretamente?
- **F1-macro:** média harmônica entre Precisão e Recall para cada classe — a métrica principal, pois trata todas as classes com o mesmo peso independente da quantidade de exemplos.

---

## Dependências

| Biblioteca | Versão mínima | Para que serve |
|---|---|---|
| torch | 2.0.0 | Framework principal de redes neurais |
| torchvision | 0.15.0 | Datasets, transformações e modelos pré-treinados |
| scikit-learn | 1.3.0 | Cálculo de F1-macro, Precisão e Recall |
| Pillow | 10.0.0 | Leitura e manipulação das imagens |

---

## Observações

- O conjunto de teste (`data/test/`) **nunca foi usado** durante o treinamento — apenas para gerar a submissão final no Kaggle.
- A validação é criada automaticamente a partir de **20% do treino** — nunca o conjunto de teste.
- O Early Stopping salva automaticamente o melhor modelo durante o treino e para quando não há mais melhora.
- O dataset não deve ser compartilhado com terceiros nem disponibilizado publicamente.

---

*Projeto desenvolvido individualmente — Mineração de Dados, UFU, 2025*
s