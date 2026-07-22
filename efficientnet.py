# =============================================================================
# efficientnet.py — EfficientNet-B0 com Transfer Learning
# =============================================================================
# Transfer Learning = pegar uma rede já treinada em milhões de imagens
# e adaptar para o nosso problema específico (maturação de café).
#
# Por que funciona tão bem com dataset pequeno?
#   A EfficientNet já aprendeu a reconhecer bordas, texturas, formas e cores
#   em 1.2 milhão de imagens (ImageNet). Para o nosso problema, ela só precisa
#   aprender a APLICAR esse conhecimento para distinguir frutos de café.
#
# Estratégia em 2 estágios:
#   Estágio 1: Congela o backbone, treina só o classificador (10 épocas)
#              → rápido, evita destruir o conhecimento pré-treinado
#   Estágio 2: Descongela tudo, fine-tuning completo com LR menor (40 épocas)
#              → ajusta os pesos para o domínio específico de café
# =============================================================================

import torch
import torch.nn as nn

# torchvision já inclui a EfficientNet com pesos pré-treinados
from torchvision import models


class EfficientNetCafe(nn.Module):
    """
    EfficientNet-B0 adaptada para classificar 5 estágios de maturação de café.

    A rede original classifica 1000 categorias do ImageNet.
    Substituímos apenas a última camada para classificar nossas 5 classes.

    Parâmetros:
        num_classes : número de classes (5)
        dropout     : taxa de dropout no classificador
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.3):
        super().__init__()

        # Carrega a EfficientNet-B0 com pesos pré-treinados no ImageNet
        # IMAGENET1K_V1 = pesos treinados em 1.2 milhão de imagens, 1000 classes
        pesos = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        self.modelo = models.efficientnet_b0(weights=pesos)

        # Descobre quantas entradas tem a última camada original
        # (EfficientNet-B0 usa 1280 features antes da classificação)
        entradas_classificador = self.modelo.classifier[1].in_features

        # Substitui o classificador original (1000 classes) pelo nosso (5 classes)
        # Mantemos um Dropout para regularização
        self.modelo.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(entradas_classificador, num_classes),
        )

    def congelar_backbone(self):
        """
        Congela todas as camadas convolucionais (backbone).
        Apenas o classificador será treinado no Estágio 1.
        Isso preserva o conhecimento pré-treinado e treina mais rápido.
        """
        for param in self.modelo.features.parameters():
            param.requires_grad = False
        print("Backbone CONGELADO — treinando apenas o classificador.")

    def descongelar_backbone(self):
        """
        Descongela todas as camadas para fine-tuning completo.
        Usado no Estágio 2 com taxa de aprendizado menor.
        """
        for param in self.modelo.parameters():
            param.requires_grad = True
        print("Backbone DESCONGELADO — fine-tuning completo.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.modelo(x)


def criar_efficientnet(num_classes: int = 5, dropout: float = 0.3) -> EfficientNetCafe:
    """
    Cria e retorna a EfficientNet-B0 pronta para transfer learning.
    """
    modelo = EfficientNetCafe(num_classes=num_classes, dropout=dropout)

    total_params = sum(p.numel() for p in modelo.parameters())
    params_treinaveis = sum(p.numel() for p in modelo.parameters()
                           if p.requires_grad)

    print("Modelo EfficientNet-B0 criado!")
    print(f"  Total de parâmetros    : {total_params:,}")
    print(f"  Parâmetros treináveis  : {params_treinaveis:,}")
    print(f"  Classificador          : 1280 → {num_classes} classes")

    return modelo
