
import csv
import argparse
from pathlib import Path

import torch

from mlp import criar_mlp
from cnn import criar_cnn
from dados import CLASSES, DatasetTeste, transformacao_avaliacao
from torch.utils.data import DataLoader


def definir_argumentos():
    parser = argparse.ArgumentParser(
        description="Gera CSV de submissão para o Kaggle"
    )
    parser.add_argument(
        "--modelo",
        type=str,
        default="cnn",
        choices=["mlp", "cnn"],
        help="Qual modelo usar: 'mlp' ou 'cnn' (padrão: cnn)"
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
    print("\nMapeamento de classes (índice numérico):")
    for idx, nome in enumerate(CLASSES):
        print(f"  {idx} = {nome}")

    # ── 1. Carrega o modelo ───────────────────────────────────────────────────
    print("\n[1/3] Carregando o modelo treinado...")

    if modelo_escolhido == "cnn":
        # CNN — melhor desempenho (F1-macro = 0.38)
        modelo = criar_cnn(
            num_classes=len(CLASSES),
            dropout=0.5,
        )
        caminho_pesos = f"{pasta_resultados}/cnn_melhor_modelo.pth"
        tamanho_imagem = 128  # CNN usa imagens 128×128

    else:
        # MLP — baseline (F1-macro = 0.27)
        modelo = criar_mlp(
            tamanho_imagem=64,
            neuronios_ocultos=[512, 256, 128],
            num_classes=len(CLASSES),
            dropout=0.5,
        )
        caminho_pesos = f"{pasta_resultados}/mlp_melhor_modelo.pth"
        tamanho_imagem = 64  # MLP usa imagens 64×64

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

    loader_teste = DataLoader(
        dataset_teste,
        batch_size=1,
        shuffle=False,  # Manter ordem dos IDs
    )

    print(f"  {len(dataset_teste)} imagens encontradas")

    # ── 3. Gera as predições ──────────────────────────────────────────────────
    print("\n[3/3] Gerando predições...")

    predicoes = []

    with torch.no_grad():
        for imagem, id_imagem in loader_teste:

            imagem = imagem.to(dispositivo)

            # Passa pela rede e obtém as pontuações para cada classe
            saidas = modelo(imagem)

            # Pega o índice da maior pontuação — este é o número que enviamos
            # Ex: 0=Cereja, 1=Passa, 2=Seco, 3=Verde, 4=Verde cana
            indice_predito = saidas.argmax(1).item()

            id_str = id_imagem[0]
            predicoes.append((id_str, indice_predito))

            # Exibe o resultado com nome e número para conferência
            nome_classe = CLASSES[indice_predito]
            print(f"  Imagem {id_str:>4} → {indice_predito} ({nome_classe})")

    # ── 4. Salva o CSV ────────────────────────────────────────────────────────
    Path(pasta_resultados).mkdir(parents=True, exist_ok=True)
    caminho_csv = f"{pasta_resultados}/submissao.csv"

    with open(caminho_csv, "w", newline="") as arquivo_csv:
        escritor = csv.writer(arquivo_csv)

        # Cabeçalho no formato do Kaggle
        escritor.writerow(["id", "class"])

        # ID como inteiro, classe como número inteiro
        for id_img, classe_num in predicoes:
            escritor.writerow([int(id_img), int(classe_num)])

    print(f"\n{'='*45}")
    print(f"{'='*45}")
    print(f"  Modelo   : {modelo_escolhido.upper()}")
    print(f"  Arquivo  : {caminho_csv}")
    print(f"  Total    : {len(predicoes)} imagens")
    print(f"{'='*45}")


if __name__ == "__main__":
    args = definir_argumentos()
    gerar_submissao(
        modelo_escolhido=args.modelo,
        pasta_teste=args.pasta_teste,
        pasta_resultados=args.pasta_resultados,
    )
