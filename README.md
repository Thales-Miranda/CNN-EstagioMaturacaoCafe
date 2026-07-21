

# Predição de Maturação de Frutos de Café

> Classificação por Imagens com Redes Neurais (MLP e CNN)  
> Disciplina: Inteligência Artificial • 2025

---

## 1. Descrição do Projeto

Este projeto implementa um sistema de classificação automática de imagens de frutos de café em cinco estágios de maturação, utilizando redes neurais desenvolvidas em Python com PyTorch.

O problema consiste em predizer o estágio de maturação a partir de fotos capturadas pelo agricultor. Para isso, foram implementadas duas arquiteturas:

- **MLP (Multilayer Perceptron)** — modelo baseline obrigatório
- **CNN (Rede Neural Convolucional)** — modelo avançado para melhoria do desempenho

---

## 2. Classes de Maturação

O dataset contém cinco estágios de maturação dos frutos de café:

| Índice | Classe | Descrição |
|--------|------------|--------------------------------------------------|
| 0 | Cereja | Fruto maduro, coloração vermelha intensa |
| 1 | Passa | Fruto em processo de secagem na planta |
| 2 | Seco | Fruto completamente seco na planta |
| 3 | Verde | Fruto imaturo, coloração verde |
| 4 | Verde cana | Fruto em transição, início de amadurecimento |

---

## 3. Estrutura do Projeto

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
│       └── *.jpg   (15 imagens sem rótulo)
├── resultados/
│   ├── mlp_melhor_modelo.pth
│   ├── cnn_melhor_modelo.pth
│   ├── mlp_metricas.json
│   ├── cnn_metricas.json
│   └── submissao.csv
├── dados.py
├── mlp.py
├── cnn.py
├── treino.py
├── treino_mlp.py
├── treino_cnn.py
├── submissao.py
└── README.md
```

---

## 4. Descrição dos Arquivos

| Arquivo | Função |
|---------------|--------------------------------------------------------------|
| `dados.py` | Carregamento, pré-processamento e Data Augmentation |
| `mlp.py` | Definição da arquitetura da Rede Neural MLP (baseline) |
| `cnn.py` | Definição da arquitetura da Rede Neural CNN (avançado) |
| `treino.py` | Loop de treinamento, early stopping e métricas |
| `treino_mlp.py` | Script principal para treinar a MLP |
| `treino_cnn.py` | Script principal para treinar a CNN |
| `submissao.py` | Gera o CSV de predições para submissão no Kaggle |

---

## 5. Instalação e Configuração

### Requisitos do Sistema

- Python 3.10 ou superior
- Sistema operacional: Windows, Linux (Fedora 30–36) ou macOS
- Mínimo 4 GB de RAM
- GPU NVIDIA (opcional, mas recomendado para treino mais rápido)

### Instalação das Dependências

Abra o terminal na pasta do projeto e execute:

```bash
# Criar ambiente virtual (recomendado)
python -m venv venv

# Ativar o ambiente virtual
# Windows:
venv\Scripts\activate

# Linux / macOS:
source venv/bin/activate

# Instalar as bibliotecas necessárias
pip install torch torchvision scikit-learn pillow
```

---

## 6. Como Executar

### 6.1 Treinar a MLP (baseline obrigatório)

```bash
python treino_mlp.py
```

Parâmetros opcionais:

```bash
python treino_mlp.py --pasta_dados ./data --epocas 200 --batch_size 7
```

### 6.2 Treinar a CNN (modelo avançado)

```bash
python treino_cnn.py
```

Parâmetros opcionais:

```bash
python treino_cnn.py --pasta_dados ./data --epocas 200 --batch_size 7
```

### 6.3 Gerar o CSV de Submissão

Após treinar os modelos, gere as predições para o conjunto de teste:

```bash
# Usando a CNN (recomendado — melhor desempenho)
python submissao.py

# Usando a MLP
python submissao.py --modelo mlp
```

O arquivo `submissao.csv` será salvo em `resultados/`.

---

## 7. Arquiteturas Implementadas

### 7.1 MLP — Multilayer Perceptron (Baseline)

A MLP recebe a imagem achatada como vetor de pixels e passa por camadas densas para classificar nas 5 classes.

| Parâmetro | Valor |
|----------------------|----------------------------------------------|
| Tamanho de entrada | 3 × 64 × 64 = 12.288 pixels |
| Camadas ocultas | 512 → 256 → 128 neurônios |
| Ativação | ReLU + BatchNorm + Dropout(0.5) |
| Parâmetros treináveis| 6.458.629 |
| Otimizador | Adam (lr=0.001, weight_decay=0.001) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Early Stopping | Paciência de 30 épocas |

### 7.2 CNN — Rede Neural Convolucional (Modelo Avançado)

A CNN preserva a estrutura espacial da imagem e detecta padrões visuais como bordas, texturas e cores dos frutos através de 4 blocos convolucionais.

| Parâmetro | Valor |
|----------------------|----------------------------------------------|
| Tamanho de entrada | 3 × 128 × 128 pixels |
| Blocos convolucionais| 4 blocos (32 → 64 → 128 → 256 filtros) |
| Cada bloco | Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm → ReLU → MaxPool |
| Classificador | 4096 → 512 → 128 → 5 classes |
| Parâmetros treináveis| 3.339.429 |
| Otimizador | Adam (lr=0.001, weight_decay=0.0001) |
| Scheduler | ReduceLROnPlateau (patience=10, factor=0.5) |
| Early Stopping | Paciência de 30 épocas |

---

## 8. Pré-processamento e Data Augmentation

Com apenas 35 imagens de treino, o Data Augmentation é essencial para evitar overfitting:

| Transformação | Descrição |
|----------------------|------------------------------------------------------|
| Resize | Redimensiona para tamanho + 20 pixels |
| RandomCrop | Recorte aleatório para o tamanho final |
| RandomHorizontalFlip | Espelhamento horizontal com 50% de chance |
| RandomVerticalFlip | Espelhamento vertical com 50% de chance |
| RandomRotation(30°) | Rotação aleatória entre -30° e +30° |
| ColorJitter | Variação de brilho, contraste, saturação e matiz |
| Normalize (ImageNet) | Normalização com média e desvio padrão do ImageNet |

---

## 9. Resultados Obtidos

Avaliação realizada no conjunto de validação (7 imagens — 20% do treino):

| Modelo | Precisão (macro) | Recall (macro) | F1-macro |
|----------------|------------------|----------------|----------|
| MLP (baseline) | 0.2000 | 0.4000 | 0.2667 |
| CNN (avançado) | 0.3333 | 0.5000 | **0.3800** |

A CNN melhorou o F1-macro em aproximadamente **43%** em relação à MLP.

---

## 10. Métricas de Avaliação

- **Precisão (macro):** média da precisão por classe — "de tudo que predi como Cereja, quantos eram Cereja de fato?"
- **Recall (macro):** média do recall por classe — "de todas as Cerejas reais, quantas identifiquei corretamente?"
- **F1-macro:** média harmônica entre Precisão e Recall — **métrica principal da avaliação**

---

## 11. Dependências

| Biblioteca | Versão mínima | Uso |
|------------|---------------|----------------------------------|
| torch | 2.0.0 | Framework de redes neurais |
| torchvision | 0.15.0 | Datasets e transformações |
| scikit-learn | 1.3.0 | Cálculo de métricas |
| Pillow | 10.0.0 | Leitura de imagens |

---

## 12. Observações Importantes

- O conjunto de teste (`data/test/`) **NÃO deve ser usado** durante o treinamento — apenas para gerar a submissão final.
- O conjunto de validação é criado automaticamente a partir de **20% do conjunto de treino**.
- Os pesos do melhor modelo são salvos automaticamente em `resultados/` durante o treino.
- O **Early Stopping** interrompe o treino automaticamente se não houver melhora por 30 épocas consecutivas.
- O dataset **não deve ser compartilhado** com terceiros nem disponibilizado publicamente.

---

