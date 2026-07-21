



# Carregamento e Preparação das Imagens



#Ler as imagens do disco
#Redimensionar para um tamanho padrão
#Aplicar transformações para aumentar a variedade
#Dividir os dados em treino e validação
#Criar os DataLoaders (quem entrega os batches para a rede durante o treino)



import os
from pathlib import Path

#torch e torchvision são as bibliotecas principais para trabalhar com imagens e redes neurais no PyTorch

import torch
from torch.utils.data import DataLoader, Dataset, random_split

#torchvision.datasets.ImageFolder lê automaticamente pastas organizadas por classe (cada subpasta = uma classe)

from torchvision import datasets, transforms

# PIL é usada para abrir imagens no Python

from PIL import Image






#Definição das classes


CLASSES = ["Cereja", "Passa", "Seco", "Verde", "Verde cana"]
NUM_CLASSES = len(CLASSES)




# Dicionário que mapeia o nome da classe para seu índice numérico Ex: "Cereja" → 0, "Passa" → 1


CLASSE_PARA_IDX = {nome: idx for idx, nome in enumerate(CLASSES)}


# Média e desvio padrão do ImageNet — usados para normalizar as imagens.
# Como nossas imagens são fotos comuns (igual ao ImageNet), esses valores funcionam bem mesmo sem calcular os dados.


MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]



#Transformações para o conjunto de TREINO (com augmentation)


#Data Augmentation = criar variações artificiais das imagens para que a rede veja "mais exemplos" e não decore as imagens originais.


def transformacao_treino(tamanho_imagem: int = 64) -> transforms.Compose:

    """
    Retorna as transformações aplicadas nas imagens de treino.
    Cada transformação é aplicada aleatoriamente a cada época,
    gerando variações diferentes da mesma imagem.
    """
    return transforms.Compose([

        # Primeiro aumentamos um pouco o tamanho para depois recortar
        # Isso garante que o corte não corte bordas pretas
        transforms.Resize((tamanho_imagem + 20, tamanho_imagem + 20)),

        # Recorte aleatório: pega uma região aleatória da imagem
        # Simula diferentes enquadramentos da planta pelo agricultor
        transforms.RandomCrop(tamanho_imagem),

        # Espelha horizontalmente com 50% de chance
        # Uma planta virada para esquerda é igual a uma virada para direita
        transforms.RandomHorizontalFlip(),

        # Espelha verticalmente com 50% de chance
        transforms.RandomVerticalFlip(),

        # Rotaciona aleatoriamente entre -30° e +30°
        # Simula diferentes ângulos de captura da foto
        transforms.RandomRotation(30),

        # Altera brilho, contraste, saturação e matiz aleatoriamente
        # Simula diferentes condições de iluminação (sol, sombra, etc.)
        transforms.ColorJitter(
            brightness=0.4,   # variação de brilho
            contrast=0.4,     # variação de contraste
            saturation=0.4,   # variação de saturação de cor
            hue=0.1           # pequena variação de tom de cor
        ),

        # Converte a imagem PIL para tensor PyTorch
        # Também normaliza os pixels de [0, 255] para [0.0, 1.0]
        transforms.ToTensor(),

        # Normaliza com média e desvio padrão do ImageNet
        # Isso coloca os valores numa faixa que facilita o aprendizado
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])



# Transformações para VALIDAÇÃO e TESTE (sem augmentation)
# Na avaliação, queremos resultados consistentes — sem aleatoriedade. Apenas redimensionamos e normalizamos.



def transformacao_avaliacao(tamanho_imagem: int = 64) -> transforms.Compose:
    """
    Retorna as transformações aplicadas nas imagens de validação e teste.
    Sem augmentation — apenas prepara a imagem para entrar na rede.
    """
    return transforms.Compose([
        # Redimensiona para o tamanho padrão
        transforms.Resize((tamanho_imagem, tamanho_imagem)),

        # Converte para tensor
        transforms.ToTensor(),

        # Normaliza igual ao treino
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])



#Dataset personalizado para o conjunto de TESTE (sem rótulos)


class DatasetTeste(Dataset):
    """
    Carrega as imagens do conjunto de teste (sem rótulos).
    Retorna a imagem transformada e o ID (nome do arquivo sem extensão).
    """

    def __init__(self, pasta_teste: str, transformacao=None):
        # Busca todos os arquivos .jpg dentro da pasta de teste
        # e ordena pelo ID numérico para facilitar a submissão
        self.caminhos = sorted(
            Path(pasta_teste).glob("*.jpg"),
            key=lambda p: int(p.stem)  # ordena pelo número do arquivo
        )
        self.transformacao = transformacao

    def __len__(self):
        # Retorna quantas imagens temos no teste
        return len(self.caminhos)

    def __getitem__(self, idx):
        # Abre a imagem pelo índice
        imagem = Image.open(self.caminhos[idx]).convert("RGB")

        # Aplica as transformações (redimensionar, normalizar, etc.)
        if self.transformacao:
            imagem = self.transformacao(imagem)

        # Retorna a imagem e o ID (nome do arquivo sem .jpg)
        # Ex: "44", "101", "381", ...
        id_imagem = self.caminhos[idx].stem
        return imagem, id_imagem




# Função principal de carregamento dos dados



def carregar_dados(pasta_raiz: str,
                   proporcao_val: float = 0.2,
                   tamanho_imagem: int = 64,
                   semente: int = 42):
    """
    Carrega e divide os dados em treino, validação e teste.

    Com 35 imagens e proporcao_val=0.2:
        - Treino    : 28 imagens (80%)
        - Validação :  7 imagens (20%)

    Parâmetros:
        pasta_raiz    : pasta que contém 'train/' e 'test/'
        proporcao_val : fração do treino usada como validação
        tamanho_imagem: tamanho para redimensionar as imagens (altura = largura)
        semente       : número para reprodutibilidade (mesma divisão toda vez)

    Retorna:
        dados_treino, dados_val, dados_teste
    """

    # Caminhos para as pastas de treino e teste
    pasta_treino = os.path.join(pasta_raiz, "train")
    pasta_teste  = os.path.join(pasta_raiz, "test")

    # ImageFolder lê automaticamente as subpastas como classes
    # Aplica augmentation nas imagens de treino
    dataset_completo = datasets.ImageFolder(
        pasta_treino,
        transform=transformacao_treino(tamanho_imagem)
    )

    # Calcula quantas imagens vão para validação
    n_val   = max(1, int(len(dataset_completo) * proporcao_val))
    n_treino = len(dataset_completo) - n_val

    # Divide aleatoriamente (mas de forma reprodutível pela semente)
    gerador = torch.Generator().manual_seed(semente)
    dados_treino, dados_val = random_split(
        dataset_completo, [n_treino, n_val], generator=gerador
    )

    # Para a validação, recriamos o dataset com transformações SEM augmentation
    # (não queremos aleatoriedade na avaliação)
    dataset_val_base = datasets.ImageFolder(
        pasta_treino,
        transform=transformacao_avaliacao(tamanho_imagem)
    )
    dados_val.dataset = dataset_val_base

    # Dataset de teste (sem rótulos)
    dados_teste = DatasetTeste(
        pasta_teste,
        transformacao=transformacao_avaliacao(tamanho_imagem)
    )

    # Exibe um resumo dos dados carregados
    print("=" * 45)
    print("DADOS CARREGADOS")
    print("=" * 45)
    print(f"  Treino    : {n_treino} imagens")
    print(f"  Validação : {n_val} imagens")
    print(f"  Teste     : {len(dados_teste)} imagens (sem rótulos)")
    print(f"  Classes   : {dataset_completo.classes}")
    print(f"  Tamanho   : {tamanho_imagem}×{tamanho_imagem} pixels")
    print("=" * 45)

    return dados_treino, dados_val, dados_teste



# Função para criar os DataLoaders


# DataLoader é o responsável por entregar os dados em "batches" (lotes) durante o treinamento. Por exemplo, se batch_size=7, ele entrega 7 imagens por vez para a rede aprender.




def criar_dataloaders(pasta_raiz: str,
                      batch_size: int = 7,
                      proporcao_val: float = 0.2,
                      tamanho_imagem: int = 64,
                      num_workers: int = 0,
                      semente: int = 42):
    """
    Cria e retorna os DataLoaders de treino, validação e teste.

    Parâmetros:
        pasta_raiz    : pasta com 'train/' e 'test/'
        batch_size    : quantas imagens são processadas por vez
                        (7 = 1 de cada classe por batch — bom para dataset pequeno)
        proporcao_val : fração do treino para validação
        tamanho_imagem: tamanho das imagens
        num_workers   : processos paralelos para carregar dados (0 = sem paralelismo)
        semente       : para reprodutibilidade

    Retorna:
        loader_treino, loader_val, loader_teste
    """

    # Carrega os datasets
    dados_treino, dados_val, dados_teste = carregar_dados(
        pasta_raiz, proporcao_val, tamanho_imagem, semente
    )

    # DataLoader de treino: shuffle=True embaralha as imagens a cada época
    # Isso é importante para que a rede não aprenda a ordem das imagens
    loader_treino = DataLoader(
        dados_treino,
        batch_size=batch_size,
        shuffle=True,          # embaralha a cada época
        num_workers=num_workers
    )

    # DataLoader de validação: shuffle=False — sempre na mesma ordem
    loader_val = DataLoader(
        dados_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    # DataLoader de teste: shuffle=False — importante para manter a ordem dos IDs
    loader_teste = DataLoader(
        dados_teste,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader_treino, loader_val, loader_teste


