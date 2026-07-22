# =============================================================================
# treino_efficientnet.py — Treinamento da EfficientNet com Transfer Learning
# =============================================================================
# Como usar:
#     python treino_efficientnet.py
# =============================================================================

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn

from efficientnet import criar_efficientnet
from dados import CLASSES, criar_dataloaders
from treino import avaliacao_final, treinar


def definir_argumentos():
    parser = argparse.ArgumentParser(
        description="Treina EfficientNet-B0 para maturação de café"
    )
    parser.add_argument("--pasta_dados",       default="./data")
    parser.add_argument("--tamanho_imagem",    type=int,   default=224,
                        help="EfficientNet foi treinada com 224×224 (padrão)")
    parser.add_argument("--batch_size",        type=int,   default=7)
    parser.add_argument("--epocas_estagio1",   type=int,   default=10,
                        help="Épocas com backbone congelado")
    parser.add_argument("--epocas_estagio2",   type=int,   default=40,
                        help="Épocas de fine-tuning completo")
    parser.add_argument("--lr_estagio1",       type=float, default=1e-3,
                        help="LR alto para treinar só o classificador")
    parser.add_argument("--lr_estagio2",       type=float, default=1e-4,
                        help="LR baixo para fine-tuning (evita destruir pesos)")
    parser.add_argument("--dropout",           type=float, default=0.3)
    parser.add_argument("--proporcao_val",     type=float, default=0.2)
    parser.add_argument("--paciencia",         type=int,   default=15)
    parser.add_argument("--pasta_resultados",  default="./resultados")
    parser.add_argument("--semente",           type=int,   default=42)
    return parser.parse_args()


def main():
    args = definir_argumentos()

    pasta_resultados = Path(args.pasta_resultados)
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado : {dispositivo}")

    # ── 1. Carrega os dados ───────────────────────────────────────────────────
    # EfficientNet usa 224×224 — tamanho que ela foi treinada no ImageNet
    print("\n[1/5] Carregando os dados...")
    loader_treino, loader_val, _ = criar_dataloaders(
        pasta_raiz=args.pasta_dados,
        batch_size=args.batch_size,
        proporcao_val=args.proporcao_val,
        tamanho_imagem=args.tamanho_imagem,
        semente=args.semente,
    )

    # ── 2. Cria o modelo ──────────────────────────────────────────────────────
    print("\n[2/5] Carregando EfficientNet-B0 pré-treinada...")
    modelo = criar_efficientnet(
        num_classes=len(CLASSES),
        dropout=args.dropout,
    )

    criterio = nn.CrossEntropyLoss()
    caminho_modelo = str(pasta_resultados / "efficientnet_melhor_modelo.pth")

    # ── 3. Estágio 1: Treina só o classificador ───────────────────────────────
    # Congela o backbone para preservar os pesos pré-treinados
    # Treina apenas as últimas camadas (o classificador que substituímos)
    print("\n[3/5] ESTÁGIO 1 — Treinando apenas o classificador...")
    print(f"      LR={args.lr_estagio1} | Épocas={args.epocas_estagio1}")
    modelo.congelar_backbone()

    otimizador1 = torch.optim.Adam(
        filter(lambda p: p.requires_grad, modelo.parameters()),
        lr=args.lr_estagio1,
    )
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador1, mode="min", patience=5, factor=0.5
    )

    treinar(
        modelo=modelo,
        loader_treino=loader_treino,
        loader_val=loader_val,
        criterio=criterio,
        otimizador=otimizador1,
        scheduler=scheduler1,
        epocas=args.epocas_estagio1,
        dispositivo=dispositivo,
        caminho_salvar=caminho_modelo,
        paciencia=args.paciencia,
    )

    # ── 4. Estágio 2: Fine-tuning completo ───────────────────────────────────
    # Descongela tudo e treina com LR muito menor
    # para ajustar os pesos sem destruir o conhecimento pré-treinado
    print("\n[4/5] ESTÁGIO 2 — Fine-tuning completo...")
    print(f"      LR={args.lr_estagio2} | Épocas={args.epocas_estagio2}")
    modelo.descongelar_backbone()

    otimizador2 = torch.optim.Adam(
        modelo.parameters(),
        lr=args.lr_estagio2,
        weight_decay=1e-4,
    )
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
        otimizador2, T_max=args.epocas_estagio2
    )

    treinar(
        modelo=modelo,
        loader_treino=loader_treino,
        loader_val=loader_val,
        criterio=criterio,
        otimizador=otimizador2,
        scheduler=scheduler2,
        epocas=args.epocas_estagio2,
        dispositivo=dispositivo,
        caminho_salvar=caminho_modelo,
        paciencia=args.paciencia,
    )

    # ── 5. Avaliação final ────────────────────────────────────────────────────
    print("\n[5/5] Avaliação final no conjunto de validação...")
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))
    metricas = avaliacao_final(modelo, loader_val, CLASSES, dispositivo)

    resultado = {
        "precisao_macro": metricas["precisao_macro"],
        "recall_macro"  : metricas["recall_macro"],
        "f1_macro"      : metricas["f1_macro"],
    }

    caminho_metricas = pasta_resultados / "efficientnet_metricas.json"
    with open(caminho_metricas, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*45}")
    print("TREINAMENTO CONCLUÍDO")
    print(f"{'='*45}")
    print(f"  Pesos salvos   : {caminho_modelo}")
    print(f"  Métricas       : {caminho_metricas}")
    print(f"  F1-macro final : {metricas['f1_macro']:.4f}")
    print(f"{'='*45}")


if __name__ == "__main__":
    main()
