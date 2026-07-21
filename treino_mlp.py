


# train_mlp.py — Script Principal de Treinamento da MLP



#Carregar as imagens do disco
#Criar o modelo MLP
#Treinar o modelo
#Avaliar no conjunto de validação
#Salvar os pesos e as métricas em ./resultados/






# argparse permite definir argumentos de linha de comando
# Ex: python train_mlp.py --epocas 300 --taxa_aprendizado 0.001
import argparse



# json para salvar as métricas em arquivo de texto estruturado
import json



from pathlib import Path  # para manipular caminhos de pastas/arquivos

import torch
import torch.nn as nn

# Importamos as partes que criamos nos outros arquivos

from mlp import criar_mlp
from dados import CLASSES, criar_dataloaders
from treino import avaliacao_final, treinar






# Definição dos argumentos de linha de comando



def definir_argumentos():
    parser = argparse.ArgumentParser(
        description="Treina a MLP para classificação de maturação de café"
    )

    # Caminho para a pasta com os dados (deve conter 'train/' e 'test/')
    parser.add_argument(
        "--pasta_dados",
        type=str,
        default="./data",
        help="Caminho para a pasta raiz dos dados (padrão: ./data)"
    )

    # Tamanho das imagens: usamos 64 para a MLP ter menos parâmetros
    # (evita overfitting com dataset pequeno)
    parser.add_argument(
        "--tamanho_imagem",
        type=int,
        default=64,
        help="Tamanho das imagens em pixels (padrão: 64)"
    )

    # Quantidade de imagens processadas por vez
    # 7 = uma imagem de cada classe por batch
    parser.add_argument(
        "--batch_size",
        type=int,
        default=7,
        help="Tamanho do batch (padrão: 7)"
    )

    # Número máximo de épocas (o early stopping pode parar antes)
    parser.add_argument(
        "--epocas",
        type=int,
        default=200,
        help="Número máximo de épocas de treino (padrão: 200)"
    )

    # Taxa de aprendizado: controla o tamanho dos passos na atualização dos pesos
    parser.add_argument(
        "--taxa_aprendizado",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado do otimizador Adam (padrão: 0.001)"
    )

    # Dropout: fração dos neurônios desligados durante o treino
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Taxa de dropout (padrão: 0.5)"
    )

    # Proporção de dados usada para validação
    parser.add_argument(
        "--proporcao_val",
        type=float,
        default=0.2,
        help="Fração do treino para validação (padrão: 0.2 = 20%%)"
    )

    # Paciência para early stopping
    parser.add_argument(
        "--paciencia",
        type=int,
        default=30,
        help="Épocas sem melhora antes de parar (padrão: 30)"
    )

    # Onde salvar os resultados
    parser.add_argument(
        "--pasta_resultados",
        type=str,
        default="./resultados",
        help="Pasta para salvar pesos e métricas (padrão: ./resultados)"
    )

    # Semente para reprodutibilidade
    parser.add_argument(
        "--semente",
        type=int,
        default=42,
        help="Semente aleatória para reprodutibilidade (padrão: 42)"
    )

    return parser.parse_args()








# Função principal




def main():
    # Lê os argumentos da linha de comando
    args = definir_argumentos()

    # Cria a pasta de resultados se não existir
    pasta_resultados = Path(args.pasta_resultados)
    pasta_resultados.mkdir(parents=True, exist_ok=True)

    # Define o dispositivo: usa GPU se disponível, senão CPU
    dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Dispositivo utilizado : {dispositivo}")

    # ─── 1. Carrega os dados ──────────────────────────────────────────────────
    print("\n[1/4] Carregando os dados...")
    loader_treino, loader_val, loader_teste = criar_dataloaders(
        pasta_raiz=args.pasta_dados,
        batch_size=args.batch_size,
        proporcao_val=args.proporcao_val,
        tamanho_imagem=args.tamanho_imagem,
        semente=args.semente,
    )

    # ─── 2. Cria o modelo MLP ─────────────────────────────────────────────────
    print("\n[2/4] Criando o modelo MLP...")
    modelo = criar_mlp(
        tamanho_imagem=args.tamanho_imagem,
        neuronios_ocultos=[512, 256, 128],  # 3 camadas ocultas
        num_classes=len(CLASSES),            # 5 classes de maturação
        dropout=args.dropout,
    )

    # ─── 3. Define critério, otimizador e scheduler ───────────────────────────
    print("\n[3/4] Configurando o treinamento...")

    # CrossEntropyLoss: função de perda padrão para classificação multiclasse
    # Compara a predição da rede com o rótulo real e calcula o "erro"
    criterio = nn.CrossEntropyLoss()

    # Adam: otimizador que ajusta os pesos para reduzir o erro
    # weight_decay é uma regularização L2 (penaliza pesos muito grandes)
    otimizador = torch.optim.Adam(
        modelo.parameters(),
        lr=args.taxa_aprendizado,
        weight_decay=1e-3
    )

    # ReduceLROnPlateau: reduz a taxa de aprendizado quando a perda para de melhorar
    # patience=10 → espera 10 épocas sem melhora antes de reduzir
    # factor=0.5  → divide a taxa por 2 quando reduz
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otimizador,
        mode="min",      # queremos MINIMIZAR a perda
        patience=10,
        factor=0.5,
    )

    # Caminho onde o melhor modelo será salvo
    caminho_modelo = str(pasta_resultados / "mlp_melhor_modelo.pth")

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

    # Carrega os pesos do melhor modelo salvo durante o treino
    modelo.load_state_dict(torch.load(caminho_modelo, map_location=dispositivo))

    metricas = avaliacao_final(modelo, loader_val, CLASSES, dispositivo)

    # ─── 6. Salva as métricas em JSON ─────────────────────────────────────────
    resultado = {
        "precisao_macro": metricas["precisao_macro"],
        "recall_macro"  : metricas["recall_macro"],
        "f1_macro"      : metricas["f1_macro"],
        "historico"     : historico,
    }

    caminho_metricas = pasta_resultados / "mlp_metricas.json"
    with open(caminho_metricas, "w") as f:
        json.dump(resultado, f, indent=2, ensure_ascii=False)

    # Resumo final
    print(f"\n{'='*45}")
    print("TREINAMENTO CONCLUÍDO")
    print(f"{'='*45}")
    print(f"  Pesos salvos em  : {caminho_modelo}")
    print(f"  Métricas salvas  : {caminho_metricas}")
    print(f"  F1-macro final   : {metricas['f1_macro']:.4f}")
    print(f"{'='*45}")


# =============================================================================
# Ponto de entrada do script
# =============================================================================
# Este bloco garante que o código só rode quando executamos diretamente
# com "python train_mlp.py", e não quando importamos o arquivo em outro lugar.
# =============================================================================

if __name__ == "__main__":
    main()




