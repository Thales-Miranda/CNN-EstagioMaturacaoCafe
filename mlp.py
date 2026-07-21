


#Importamos o PyTorch, que é a biblioteca que nos permite criar redes neurais

import torch

# torch.nn contém os blocos prontos para montar redes neurais
# (camadas, funções de ativação, etc.)



import torch.nn as nn



class MLP(nn.Module):

    def __init__(self,
                 tamanho_entrada: int,
                 neuronios_ocultos: list = [512, 256, 128],
                 num_classes: int = 5,
                 dropout: float = 0.5):

        # Sempre precisamos chamar o __init__ da classe pai (nn.Module)
        super().__init__()

        # Vamos construir as camadas da rede dinamicamente
        # para facilitar testar diferentes arquiteturas
        camadas = []

        # "prev" guarda o tamanho da camada anterior
        # Começa com o tamanho da entrada (os pixels da imagem)
        prev = tamanho_entrada

        # Para cada tamanho de camada oculta que definimos, criamos um bloco
        for n_neuronios in neuronios_ocultos:

            # --- Camada Linear (Densa) ---
            # É a camada principal da MLP: cada neurônio desta camada se
            # conecta com TODOS os neurônios da camada anterior.
            # Parâmetros: (entradas, saídas)
            camadas.append(nn.Linear(prev, n_neuronios))

            # --- Batch Normalization ---
            # Normaliza os valores que saem da camada linear para ficarem
            # numa faixa parecida. Isso ajuda o treino a ser mais estável
            # e mais rápido. Pense como "organizar" os dados antes de passar
            # para o próximo passo.
            camadas.append(nn.BatchNorm1d(n_neuronios))

            # --- Função de Ativação ReLU ---
            # Sem funções de ativação, empilhar camadas lineares não teria
            # utilidade — seria equivalente a uma só camada.
            # ReLU é simples: se o valor for negativo, vira 0. Se positivo,
            # permanece igual. Isso adiciona "não-linearidade" à rede.
            # inplace=True economiza memória ao modificar o tensor diretamente.
            camadas.append(nn.ReLU(inplace=True))

            # --- Dropout ---
            # Durante o treino, desliga aleatoriamente uma fração dos neurônios
            # (definida pelo parâmetro dropout, ex: 0.5 = 50%).
            # Isso força a rede a aprender padrões mais robustos, evitando
            # que ela simplesmente "decore" as poucas imagens que temos.
            # Durante a avaliação, o Dropout é desativado automaticamente.
            camadas.append(nn.Dropout(dropout))

            # Atualiza o "prev" para o tamanho desta camada
            prev = n_neuronios

        # --- Camada de Saída ---
        # Última camada: transforma os neurônios ocultos nas 5 classes.
        # Não tem ativação aqui — o PyTorch aplica isso internamente
        # durante o cálculo da função de perda (CrossEntropyLoss).
        camadas.append(nn.Linear(prev, num_classes))

        # nn.Sequential "empacota" todas as camadas em ordem,
        # formando o pipeline completo da rede.
        self.rede = nn.Sequential(*camadas)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Define como os dados percorrem a rede (passagem para frente).

        O PyTorch chama este método automaticamente quando fazemos:
            saida = modelo(imagem)

        Parâmetro:
            x : tensor com formato (B, C, H, W)
                B = número de imagens no batch
                C = canais de cor (3 para RGB)
                H = altura da imagem
                W = largura da imagem

        Retorna:
            Tensor com formato (B, 5) — uma pontuação por classe para
            cada imagem. A classe com maior pontuação é a predição.
        """

        # A MLP não entende imagens 2D — ela precisa de um vetor 1D.
        # Então "achatamos" a imagem: (B, C, H, W) → (B, C*H*W)
        # x.size(0) pega o número de imagens (B) para manter o batch intacto.
        x = x.view(x.size(0), -1)

        # Agora passamos o vetor pela rede que construímos
        return self.rede(x)





# Função auxiliar para criar a MLP facilmente




def criar_mlp(tamanho_imagem: int = 64,
              neuronios_ocultos: list = None,
              num_classes: int = 5,
              dropout: float = 0.5) -> MLP:
    """
    Cria e retorna uma MLP pronta para uso.

    Parâmetros:
        tamanho_imagem   : altura (e largura) da imagem após redimensionamento
        neuronios_ocultos: arquitetura das camadas ocultas
        num_classes      : número de classes (5 para o nosso dataset)
        dropout          : taxa de dropout para regularização
    """

    # Se não passarmos uma arquitetura, usamos esse padrão enxuto

    if neuronios_ocultos is None:
        neuronios_ocultos = [512, 256, 128]

    # Calculamos o tamanho de entrada: 3 canais RGB × altura × largura
    tamanho_entrada = 3 * tamanho_imagem * tamanho_imagem

    # Criamos o modelo
    modelo = MLP(tamanho_entrada, neuronios_ocultos, num_classes, dropout)



    # Contamos e exibimos o total de parâmetros treináveis



    total_params = sum(p.numel() for p in modelo.parameters() if p.requires_grad)
    print(f"Modelo MLP criado!")
    print(f"  Entrada        : {tamanho_entrada} pixels (3 × {tamanho_imagem} × {tamanho_imagem})")
    print(f"  Camadas ocultas: {neuronios_ocultos}")
    print(f"  Saída          : {num_classes} classes")
    print(f"  Parâmetros     : {total_params:,}")

    return modelo


