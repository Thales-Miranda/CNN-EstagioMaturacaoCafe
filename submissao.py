# =============================================================================
# submissao.py — Gera o CSV de Submissão para o Kaggle
# =============================================================================
# Usa o melhor modelo treinado (EfficientNet por padrão) para gerar as
# predições das 15 imagens do conjunto de teste (sem rótulos).
#
# A coluna 'class' é enviada como NÚMERO conforme o mapeamento:
#     Verde=0, Verde cana=1, Cereja=2, Passa=3, Seco=4
#
# Como usar:
#     python submissao.py                    ← usa EfficientNet (recomendado)
#     python submissao.py --modelo cnn       ← usa a CNN
#     python submissao.py --modelo mlp       ← usa a MLP
#
# Resultado:
#     resultados/submissao.csv  ← arquivo para enviar no Kaggle
# =============================================================================

import csv
import argparse
from pathlib import Path

import torch

from mlp import criar_mlp
from cnn import criar_cnn
from efficientnet import criar_efficientnet
from dados import CLASSES, CLASSE_PARA_IDX, DatasetTeste, transformacao_avaliacao
from torch.utils.data import DataLoader


def definir_argumentos():
    parser = argparse.ArgumentParser(
        description="Gera CSV de submissão para o Kaggle"
    )

    # EfficientNet é o padrão pois teve melhor desempenho (F1=0.89)
    parser.add_argument(
        "--modelo",
        type=str,
        default="efficientnet",
        choices=["mlp", "cnn", "efficientnet"],
        help="Modelo para predição: 'mlp', 'cnn' ou 'efficientnet' (padrão: efficientnet)"
    )
    parser.add_argument(
        "--pasta_teste",
        type=str,
        default="./data/test",
        help="Pasta com as imagens de teste (padrão: ./data/test)"
    )
    parser.add_argument(
        "--pasta_resultados",
        type=str,
        default="./resultados",
        help="Pasta com os pesos e onde salvar o CSV (padrão: ./resultados)"
    )
    return parser.parse_args()


def gerar_submissao(modelo_escolhido, pasta_teste, pasta_resultados):

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo    : {dispositivo}")
    print(f"Modelo         : {modelo_escolhido.upper()}")

    # Exibe o mapeamento para conferência
    print("\nMapeamento de classes:")
    for nome, idx in CLASSE_PARA_IDX.items():
        print(f"  {idx} = {nome}")

    # ── 1. Carrega o modelo e os pesos ───────────────────────────────────────
    print("\n[1/3] Carregando o modelo treinado...")

    if modelo_escolhido == "efficientnet":
        # EfficientNet — melhor desempenho (F1-macro = 0.89)
        modelo = criar_efficientnet(
            num_classes=len(CLASSES),
            dropout=0.3,
        )
        caminho_pesos = f"{pasta_resultados}/efficientnet_melhor_modelo.pth"
        tamanho_imagem = 224

    elif modelo_escolhido == "cnn":
        # CNN — segundo melhor (F1-macro = 0.63)
        modelo = criar_cnn(
            num_classes=len(CLASSES),
            dropout=0.5,
        )
        caminho_pesos = f"{pasta_resultados}/cnn_melhor_modelo.pth"
        tamanho_imagem = 128

    else:
        # MLP — baseline (F1-macro = 0.27)
        modelo = criar_mlp(
            tamanho_imagem=64,
            neuronios_ocultos=[512, 256, 128],
            num_classes=len(CLASSES),
            dropout=0.5,
        )
        caminho_pesos = f"{pasta_resultados}/mlp_melhor_modelo.pth"
        tamanho_imagem = 64

    # Carrega os pesos do melhor modelo salvo durante o treino
    modelo.load_state_dict(
        torch.load(caminho_pesos, map_location=dispositivo)
    )

    # Modo de avaliação: desativa Dropout
    modelo.eval()
    modelo.to(dispositivo)
    print(f"  Pesos carregados: {caminho_pesos}")

    # ── 2. Carrega as imagens de teste ────────────────────────────────────────
    print("\n[2/3] Carregando imagens de teste...")

    dataset_teste = DatasetTeste(
        pasta_teste=pasta_teste,
        transformacao=transformacao_avaliacao(tamanho_imagem)
    )

    loader_teste = DataLoader(dataset_teste, batch_size=1, shuffle=False)
    print(f"  {len(dataset_teste)} imagens encontradas")

    # ── 3. Gera as predições ──────────────────────────────────────────────────
    print("\n[3/3] Gerando predições...")

    predicoes = []

    with torch.no_grad():
        for imagem, id_imagem in loader_teste:
            imagem = imagem.to(dispositivo)

            # Passa pela rede e obtém as pontuações para cada classe
            saidas = modelo(imagem)

            # Índice da classe predita (já está na nossa ordem correta)
            # Verde=0, Verde cana=1, Cereja=2, Passa=3, Seco=4
            indice_predito = saidas.argmax(1).item()

            # Nome da classe para exibição no terminal
            nome_classe = CLASSES[indice_predito]

            id_str = id_imagem[0]
            predicoes.append((id_str, indice_predito))
            print(f"  Imagem {id_str:>4} → {indice_predito} ({nome_classe})")

    # ── 4. Salva o CSV ────────────────────────────────────────────────────────
    Path(pasta_resultados).mkdir(parents=True, exist_ok=True)
    caminho_csv = f"{pasta_resultados}/submissao.csv"

    with open(caminho_csv, "w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)

        # Cabeçalho no formato do Kaggle
        escritor.writerow(["id", "class"])

        # ID como inteiro, classe como NÚMERO inteiro
        for id_img, classe_num in predicoes:
            escritor.writerow([int(id_img), int(classe_num)])

    print(f"\n{'='*45}")
    print("SUBMISSÃO GERADA COM SUCESSO!")
    print(f"{'='*45}")
    print(f"  Modelo   : {modelo_escolhido.upper()}")
    print(f"  Arquivo  : {caminho_csv}")
    print(f"  Total    : {len(predicoes)} imagens")
    print(f"{'='*45}")
    print("\nEnvie o arquivo 'submissao.csv' no Kaggle!")


if __name__ == "__main__":
    args = definir_argumentos()
    gerar_submissao(
        modelo_escolhido=args.modelo,
        pasta_teste=args.pasta_teste,
        pasta_resultados=args.pasta_resultados,
    )
