



# CNN Rede Neural Convolucional - Potencialização de acerto classificação 


import torch
import torch.nn as nn



#"bloco" reutilizável que agrupa as operações que se repetem em cada camada convolucional

class BlocoConvolucional(nn.Module):
     
   
    def __init__(self, canais_entrada: int, canais_saida: int):
        super().__init__()

        self.bloco = nn.Sequential(

            # --- Primeira Convolução ---
            # Conv2d aplica filtros (kernels) que deslizam sobre a imagem
            # detectando padrões como bordas, texturas e formas.
            # kernel_size=3 → filtro 3×3 pixels
            # padding=1     → adiciona borda de zeros para manter o tamanho
            nn.Conv2d(canais_entrada, canais_saida,
                      kernel_size=3, padding=1),

            # Normaliza os valores entre as camadas para treino mais estável
            nn.BatchNorm2d(canais_saida),

            # Ativação ReLU: zera valores negativos, mantém positivos
            nn.ReLU(inplace=True),

            # --- Segunda Convolução ---
            # Uma segunda convolução no mesmo bloco permite aprender
            # padrões mais complexos antes de reduzir o tamanho
            nn.Conv2d(canais_saida, canais_saida,
                      kernel_size=3, padding=1),

            nn.BatchNorm2d(canais_saida),

            nn.ReLU(inplace=True),

            # --- MaxPooling ---
            # Reduz o tamanho espacial pela metade (divide altura e largura por 2)
            # kernel_size=2, stride=2 → janela 2×2, passo de 2
            # Pega o valor MÁXIMO de cada janela 2×2 — preserva o padrão mais forte
            # Isso também reduz o número de parâmetros e ajuda a evitar overfitting
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bloco(x)








# Arquitetura completa da CNN



class CNN(nn.Module):
    """
    CNN com 4 blocos convolucionais seguidos de um classificador denso.

    Fluxo de dados com imagem 128×128:
        Entrada : (B,   3, 128, 128)  ← 3 canais RGB
        Bloco 1 : (B,  32,  64,  64)  ← 32 filtros, tamanho reduzido pela metade
        Bloco 2 : (B,  64,  32,  32)  ← 64 filtros
        Bloco 3 : (B, 128,  16,  16)  ← 128 filtros
        Bloco 4 : (B, 256,   8,   8)  ← 256 filtros
        AvgPool : (B, 256,   4,   4)  ← reduz para 4×4
        Flatten : (B, 4096)           ← achata para vetor
        Linear  : (B, 512)            ← camada densa
        Saída   : (B,   5)            ← 5 classes

    Parâmetros:
        num_classes : número de classes (5)
        dropout     : taxa de dropout no classificador
    """

    def __init__(self, num_classes: int = 5, dropout: float = 0.5):
        super().__init__()

        # --- Extrator de características (blocos convolucionais) ---
        # Cada bloco dobra o número de filtros e reduz o tamanho pela metade
        # Isso é um padrão clássico em CNNs: mais filtros conforme diminui espacialmente
        self.extrator = nn.Sequential(
            BlocoConvolucional(3,   32),   # RGB → 32 filtros  | 128→64
            BlocoConvolucional(32,  64),   # 32  → 64 filtros  |  64→32
            BlocoConvolucional(64,  128),  # 64  → 128 filtros |  32→16
            BlocoConvolucional(128, 256),  # 128 → 256 filtros |  16→8
        )

        # --- Pooling adaptativo ---
        # Independente do tamanho de entrada, sempre gera saída 4×4
        # Isso torna a CNN flexível para diferentes tamanhos de imagem
        self.pooling = nn.AdaptiveAvgPool2d((4, 4))

        # --- Classificador (igual à MLP) ---
        # Recebe o vetor achatado e classifica nas 5 classes
        self.classificador = nn.Sequential(

            # Achata o tensor 3D (256, 4, 4) em vetor 1D (4096)
            nn.Flatten(),

            # Primeira camada densa
            nn.Linear(256 * 4 * 4, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),

            # Segunda camada densa
            nn.Linear(512, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout / 2),  # dropout menor perto da saída

            # Camada de saída: 5 classes
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Passagem dos dados pela rede.

        Parâmetro:
            x : tensor (B, 3, H, W) — batch de imagens RGB

        Retorna:
            tensor (B, 5) — pontuação para cada classe
        """

        # 1. Extrai padrões visuais com as convoluções
        x = self.extrator(x)

        # 2. Reduz para tamanho fixo 4×4
        x = self.pooling(x)

        # 3. Classifica nas 5 classes
        x = self.classificador(x)

        return x
    

# Função auxiliar para criar a CNN facilmente



def criar_cnn(num_classes: int = 5, dropout: float = 0.5) -> CNN:
    """
    Cria e retorna uma CNN pronta para uso.

    Parâmetros:
        num_classes : número de classes (5 para o nosso dataset)
        dropout     : taxa de dropout para regularização
    """

    modelo = CNN(num_classes=num_classes, dropout=dropout)

    # Conta os parâmetros treináveis
    total_params = sum(p.numel() for p in modelo.parameters()
                       if p.requires_grad)

    print("Modelo CNN criado!")
    print(f"  Blocos conv    : 4 (3→32→64→128→256 filtros)")
    print(f"  Classificador  : 4096 → 512 → 128 → {num_classes}")
    print(f"  Dropout        : {dropout}")
    print(f"  Parâmetros     : {total_params:,}")

    return modelo








