



# Principal de Treinamento da CNN




#Carregar as imagens do disco
#Criar o modelo CNN
#Treinar o modelo
#Avaliar no conjunto de validação
#Salvar os pesos e as métricas em ./resultados/





import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn


# Importa a CNN 

from cnn import criar_cnn
from dados import CLASSES, criar_dataloaders
from treino import avaliacao_final, treinar


def definir_argumentos():
    parser = argparse.ArgumentParser(
        description="Treina a CNN para classificação de maturação de café"
    )

    parser.add_argument(
        "--pasta_dados",
        type=str,
        default="./data",
        help="Caminho para a pasta raiz dos dados (padrão: ./data)"
    )


    # CNN usa imagens maiores que a MLP pois consegue aproveitar
    # melhor os detalhes espaciais (texturas, formas, cores)
    parser.add_argument(
        "--tamanho_imagem",
        type=int,
        default=128,
        help="Tamanho das imagens em pixels (padrão: 128)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=7,
        help="Tamanho do batch (padrão: 7)"
    )

    parser.add_argument(
        "--epocas",
        type=int,
        default=200,
        help="Número máximo de épocas de treino (padrão: 200)"
    )

    # CNN geralmente precisa de taxa de aprendizado um pouco menor
    parser.add_argument(
        "--taxa_aprendizado",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado do otimizador Adam (padrão: 0.001)"
    )

    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Taxa de dropout (padrão: 0.5)"
    )

    parser.add_argument(
        "--proporcao_val",
        type=float,
        default=0.2,
        help="Fração do treino para validação (padrão: 0.2)"
    )

    parser.add_argument(
        "--paciencia",
        type=int,
        default=30,
        help="Épocas sem melhora antes de parar (padrão: 30)"
    )

    parser.add_argument(
        "--pasta_resultados",
        type=str,
        default="./resultados",
        help="Pasta para salvar pesos e métricas (padrão: ./resultados)"
    )

    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (padrão: 42)"
    )

    return parser.parse_args()


def main():
    args = definir_argumentos()

    pasta_resultados = Path(args.pasta_resultados)
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado : {dispositivo}")

    # ─── 1. Carrega os dados ──────────────────────────────────────────────────
    # CNN usa imagem 128×128 — maior que a MLP (64×64)
    # pois consegue aproveitar melhor os detalhes visuais
    print("\n[1/4] Carregando os dados...")
    loader_treino, loader_val, loader_teste = criar_dataloaders(
        pasta_raiz=args.pasta_dados,
        batch_size=args.batch_size,
        proporcao_val=args.proporcao_val,
        tamanho_imagem=args.tamanho_imagem,
        semente=args.semente,
    )

    # ─── 2. Cria o modelo CNN ─────────────────────────────────────────────────
    print("\n[2/4] Criando o modelo CNN...")
    modelo = criar_cnn(
        num_classes=len(CLASSES),
        dropout=args.dropout,
    )

    # ─── 3. Define critério, otimizador e scheduler ───────────────────────────
    print("\n[3/4] Configurando o treinamento...")

    # Mesma função de perda da MLP — padrão para classificação multiclasse
    criterio = nn.CrossEntropyLoss()

    # Adam com weight_decay para regularização
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=args.taxa_aprendizado,
        weight_decay=1e-4   # menor que na MLP pois a CNN já tem mais regularização
    )

    # Scheduler: reduz a taxa de aprendizado quando a perda estagna
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador,
        mode="min",
        patience=10,
        factor=0.5,
    )

    caminho_modelo = str(pasta_resultados / "cnn_melhor_modelo.pth")

    # ─── 4. Treina o modelo ───────────────────────────────────────────────────
    print("\n[4/4] Iniciando o treinamento...\n")
    historico = treinar(
        modelo=modelo,
        loader_treino=loader_treino,
        loader_val=loader_val,
        criterio=criterio,
        otimizador=otimizador,
        scheduler=scheduler,
        epocas=args.epocas,
        dispositivo=dispositivo,
        caminho_salvar=caminho_modelo,
        paciencia=args.paciencia,
    )

    # ─── 5. Avaliação final ───────────────────────────────────────────────────
    print("\n── Avaliação final no conjunto de validação ──")
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))
    metricas = avaliacao_final(modelo, loader_val, CLASSES, dispositivo)

    # ─── 6. Salva as métricas ─────────────────────────────────────────────────
    resultado = {
        "precisao_macro": metricas["precisao_macro"],
        "recall_macro"  : metricas["recall_macro"],
        "f1_macro"      : metricas["f1_macro"],
        "historico"     : historico,
    }

    caminho_metricas = pasta_resultados / "cnn_metricas.json"
    with open(caminho_metricas, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*45}")
    print("TREINAMENTO CONCLUÍDO")
    print(f"{'='*45}")
    print(f"  Pesos salvos em  : {caminho_modelo}")
    print(f"  Métricas salvas  : {caminho_metricas}")
    print(f"  F1-macro final   : {metricas['f1_macro']:.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()


