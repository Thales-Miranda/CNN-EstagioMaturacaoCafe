# Carregamento e Preparação das Imagens

# Ler as imagens do disco
# Redimensionar para um tamanho padrão
# Aplicar transformações para aumentar a variedade
# Dividir os dados em treino e validação
# Criar os DataLoaders (quem entrega os batches para a rede durante o treino)

import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms
from PIL import Image


# =============================================================================
# Definição das classes NA ORDEM CORRETA do projeto
# =============================================================================
# Verde=0, Verde cana=1, Cereja=2, Passa=3, Seco=4
# =============================================================================

CLASSES = ["Verde", "Verde cana", "Cereja", "Passa", "Seco"]
NUM_CLASSES = len(CLASSES)

# Dicionário que mapeia o nome da classe para seu índice numérico
# Ex: "Verde" → 0, "Verde cana" → 1, "Cereja" → 2, etc.
CLASSE_PARA_IDX = {nome: idx for idx, nome in enumerate(CLASSES)}

# Média e desvio padrão do ImageNet — usados para normalizar as imagens.
MEDIA_IMAGENET = [0.485, 0.456, 0.406]
DESVIO_IMAGENET = [0.229, 0.224, 0.225]


def transformacao_treino(tamanho_imagem: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((tamanho_imagem + 20, tamanho_imagem + 20)),
        transforms.RandomCrop(tamanho_imagem),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.4, contrast=0.4,
                               saturation=0.4, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def transformacao_avaliacao(tamanho_imagem: int = 64) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((tamanho_imagem, tamanho_imagem)),
        transforms.ToTensor(),
        transforms.Normalize(MEDIA_IMAGENET, DESVIO_IMAGENET),
    ])


def aplicar_ordem_classes(dataset):
    """
    Corrige o mapeamento do ImageFolder para usar nossa ordem de classes.
    Salva as classes ORIGINAIS antes de sobrescrever para não perder a referência.
    """
    # Guarda o mapeamento original (alfabético) ANTES de sobrescrever
    classes_originais = dataset.classes  # ex: ['Cereja', 'Passa', 'Seco', 'Verde', 'Verde cana']
    idx_original = dataset.class_to_idx  # ex: {'Cereja': 0, 'Passa': 1, ...}

    # Sobrescreve com nossa ordem correta
    dataset.classes = CLASSES
    dataset.class_to_idx = CLASSE_PARA_IDX

    # Reatribui os rótulos usando o mapeamento original como ponte
    # Ex: arquivo era label=0 (Cereja na ordem antiga) → vira label=2 (Cereja na nova ordem)
    dataset.samples = [
        (path, CLASSE_PARA_IDX[classes_originais[label]])
        for path, label in dataset.samples
    ]
    dataset.targets = [s[1] for s in dataset.samples]

    return dataset


class DatasetTeste(Dataset):
    """
    Carrega as imagens do conjunto de teste (sem rótulos).
    Retorna a imagem transformada e o ID (nome do arquivo sem extensão).
    """

    def __init__(self, pasta_teste: str, transformacao=None):
        self.caminhos = sorted(
            Path(pasta_teste).glob("*.jpg"),
            key=lambda p: int(p.stem)
        )
        self.transformacao = transformacao

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        imagem = Image.open(self.caminhos[idx]).convert("RGB")
        if self.transformacao:
            imagem = self.transformacao(imagem)
        id_imagem = self.caminhos[idx].stem
        return imagem, id_imagem


def carregar_dados(pasta_raiz: str,
                   proporcao_val: float = 0.2,
                   tamanho_imagem: int = 64,
                   semente: int = 42):
    """
    Carrega e divide os dados em treino, validação e teste.

    Com 35 imagens e proporcao_val=0.2:
        - Treino    : 28 imagens (80%)
        - Validação :  7 imagens (20%)
    """

    pasta_treino = os.path.join(pasta_raiz, "train")
    pasta_teste  = os.path.join(pasta_raiz, "test")

    # Carrega o dataset de treino com augmentation
    dataset_completo = datasets.ImageFolder(
        pasta_treino,
        transform=transformacao_treino(tamanho_imagem)
    )

    # Corrige a ordem das classes ANTES de dividir
    dataset_completo = aplicar_ordem_classes(dataset_completo)

    n_val    = max(1, int(len(dataset_completo) * proporcao_val))
    n_treino = len(dataset_completo) - n_val

    gerador = torch.Generator().manual_seed(semente)
    dados_treino, dados_val = random_split(
        dataset_completo, [n_treino, n_val], generator=gerador
    )

    # Validação sem augmentation — recria o dataset com transforms corretos
    dataset_val_base = datasets.ImageFolder(
        pasta_treino,
        transform=transformacao_avaliacao(tamanho_imagem)
    )
    dataset_val_base = aplicar_ordem_classes(dataset_val_base)
    dados_val.dataset = dataset_val_base

    # Dataset de teste (sem rótulos)
    dados_teste = DatasetTeste(
        pasta_teste,
        transformacao=transformacao_avaliacao(tamanho_imagem)
    )

    print("=" * 45)
    print("DADOS CARREGADOS")
    print("=" * 45)
    print(f"  Treino    : {n_treino} imagens")
    print(f"  Validação : {n_val} imagens")
    print(f"  Teste     : {len(dados_teste)} imagens (sem rótulos)")
    print(f"  Classes   : {CLASSES}")
    print(f"  Tamanho   : {tamanho_imagem}×{tamanho_imagem} pixels")
    print("=" * 45)

    return dados_treino, dados_val, dados_teste


def criar_dataloaders(pasta_raiz: str,
                      batch_size: int = 7,
                      proporcao_val: float = 0.2,
                      tamanho_imagem: int = 64,
                      num_workers: int = 0,
                      semente: int = 42):
    """
    Cria e retorna os DataLoaders de treino, validação e teste.
    """
    dados_treino, dados_val, dados_teste = carregar_dados(
        pasta_raiz, proporcao_val, tamanho_imagem, semente
    )

    loader_treino = DataLoader(
        dados_treino, batch_size=batch_size,
        shuffle=True, num_workers=num_workers
    )
    loader_val = DataLoader(
        dados_val, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )
    loader_teste = DataLoader(
        dados_teste, batch_size=batch_size,
        shuffle=False, num_workers=num_workers
    )

    return loader_treino, loader_val, loader_teste
