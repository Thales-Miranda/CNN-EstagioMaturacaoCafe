

# Treinamento e Avaliação da Rede Neural


#Treinar a rede por uma época (train_uma_epoca)
#Avaliar a rede no conjunto de validação (avaliar)
#Controlar o loop completo de treinamento com early stopping (treinar)
#Gerar as métricas finais (avaliacao_final)






import time  # para medir o tempo de cada época



import torch
import torch.nn as nn
from torch.utils.data import DataLoader



# sklearn (scikit-learn) tem funções prontas para calcular as métricas ue o projeto pede: Precisão, Recall e F1-macro


from sklearn.metrics import (
    f1_score,
    precision_score,
    recall_score,
    classification_report,  # relatório completo por classe
    confusion_matrix         # tabela de acertos e erros por classe
)



# Função 1: Treinar por uma época


#A rede faz uma predição (forward pass)
#Calculamos o erro (loss)
#Propagamos o erro de volta (backward pass)
#Atualizamos os pesos (optimizer.step)




def treinar_uma_epoca(modelo, loader, criterio, otimizador, dispositivo):
 
    # Coloca o modelo em modo de treino
    # Isso ativa o Dropout e o BatchNorm no modo correto
    modelo.train()

    perda_total = 0.0  # acumula a perda de todos os batches
    acertos     = 0    # conta quantas predições foram corretas
    total       = 0    # conta o total de imagens processadas

    # Percorre cada batch de imagens
    for imagens, rotulos in loader:

        # Move os dados para o dispositivo correto (CPU ou GPU)
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        # Zera os gradientes do passo anterior
        # (se não zeramos, os gradientes se acumulam incorretamente)
        otimizador.zero_grad()

        # FORWARD PASS: passa as imagens pela rede e obtém as predições
        # saidas tem formato (batch_size, 5) — uma pontuação por classe
        saidas = modelo(imagens)

        # Calcula a perda: o quanto a predição errou em relação ao rótulo real
        # CrossEntropyLoss compara as pontuações com o rótulo correto
        perda = criterio(saidas, rotulos)

        # BACKWARD PASS: calcula os gradientes (quanto cada peso contribuiu pro erro)
        perda.backward()

        # Atualiza os pesos com base nos gradientes calculados
        otimizador.step()

        # Acumula as estatísticas do batch
        # .item() converte tensor para número Python
        perda_total += perda.item() * imagens.size(0)

        # argmax(1) pega o índice da maior pontuação (= classe predita)
        classe_predita = saidas.argmax(1)
        acertos += (classe_predita == rotulos).sum().item()
        total   += imagens.size(0)

    # Calcula médias sobre todas as imagens
    perda_media = perda_total / total
    acuracia    = acertos / total

    return perda_media, acuracia




#Avaliar no conjunto de validação


@torch.no_grad()
def avaliar(modelo, loader, criterio, dispositivo):
    """
    Avalia o modelo sem atualizar os pesos.

    Retorna:
        perda_media, acuracia
    """

    # Coloca o modelo em modo de avaliação
    # Isso desativa o Dropout e ajusta o BatchNorm
    modelo.eval()

    perda_total = 0.0
    acertos     = 0
    total       = 0

    for imagens, rotulos in loader:
        imagens = imagens.to(dispositivo)
        rotulos = rotulos.to(dispositivo)

        # Só forward pass — sem backward, sem atualização
        saidas = modelo(imagens)
        perda  = criterio(saidas, rotulos)

        perda_total += perda.item() * imagens.size(0)
        acertos     += (saidas.argmax(1) == rotulos).sum().item()
        total       += imagens.size(0)

    return perda_total / total, acertos / total




#Loop completo de treinamento



# Aqui controlamos quantas épocas treinar, salvamos o melhor modelo,
# e implementamos o "early stopping" — parar o treino se a rede parar
# de melhorar, para evitar desperdício de tempo e overfitting.



def treinar(modelo, loader_treino, loader_val,
            criterio, otimizador, scheduler=None,
            epocas=200, dispositivo=None,
            caminho_salvar="melhor_modelo.pth",
            paciencia=30):
    """
    Executa o loop completo de treinamento.

    Early Stopping: se a perda de validação não melhorar por 'paciencia'
    épocas consecutivas, o treino é encerrado automaticamente.

    Parâmetros:
        modelo         : rede neural a ser treinada
        loader_treino  : DataLoader de treino
        loader_val     : DataLoader de validação
        criterio       : função de perda
        otimizador     : otimizador (Adam)
        scheduler      : agendador de taxa de aprendizado (opcional)
        epocas         : número máximo de épocas
        dispositivo    : 'cpu' ou 'cuda'
        caminho_salvar : onde salvar os pesos do melhor modelo
        paciencia      : épocas sem melhora para parar o treino

    Retorna:
        historico : dicionário com as métricas de cada época
    """

    # Se não especificamos o dispositivo, usa GPU se disponível
    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Move o modelo para o dispositivo
    modelo.to(dispositivo)

    # Dicionário para guardar o histórico de métricas
    historico = {
        "perda_treino": [],
        "acuracia_treino": [],
        "perda_val": [],
        "acuracia_val": []
    }

    # Variáveis para o early stopping
    melhor_perda_val   = float("inf")  # começa com infinito (qualquer valor melhora)
    epocas_sem_melhora = 0             # contador de épocas sem melhora

    print(f"\nIniciando treinamento no dispositivo: {dispositivo}")
    print(f"Épocas máximas: {epocas} | Paciência: {paciencia}\n")

    for epoca in range(1, epocas + 1):

        # Marca o tempo de início da época
        inicio = time.time()

        # Treina por uma época e obtém as métricas
        perda_tr, acc_tr = treinar_uma_epoca(
            modelo, loader_treino, criterio, otimizador, dispositivo
        )

        # Avalia no conjunto de validação
        perda_val, acc_val = avaliar(
            modelo, loader_val, criterio, dispositivo
        )

        # Atualiza o scheduler (se existir)
        # ReduceLROnPlateau precisa receber a perda de validação
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(perda_val)
            else:
                scheduler.step()

        # Guarda as métricas no histórico
        historico["perda_treino"].append(perda_tr)
        historico["acuracia_treino"].append(acc_tr)
        historico["perda_val"].append(perda_val)
        historico["acuracia_val"].append(acc_val)

        # Tempo gasto na época
        tempo = time.time() - inicio

        # Exibe o progresso
        print(f"[{epoca:03d}/{epocas}] "
              f"treino: perda={perda_tr:.4f} acc={acc_tr:.4f} | "
              f"val: perda={perda_val:.4f} acc={acc_val:.4f} | "
              f"{tempo:.1f}s")

        # Verifica se o modelo melhorou
        if perda_val < melhor_perda_val:
            # Novo melhor modelo! Salva os pesos no disco
            melhor_perda_val   = perda_val
            epocas_sem_melhora = 0
            torch.save(modelo.state_dict(), caminho_salvar)
            print(f"  ✓ Melhor modelo salvo! (perda_val={perda_val:.4f})")
        else:
            # Não melhorou — incrementa o contador
            epocas_sem_melhora += 1
            if epocas_sem_melhora >= paciencia:
                print(f"\n⚠ Early stopping: {paciencia} épocas sem melhora.")
                print(f"  Melhor perda de validação: {melhor_perda_val:.4f}")
                break

    return historico






#Avaliação final com métricas completas


# Usada para avaliar no conjunto de VALIDAÇÃO com os rótulos disponíveis.
# Calcula Precisão, Recall e F1-macro — as métricas do projeto.




@torch.no_grad()
def avaliacao_final(modelo, loader_val, nomes_classes, dispositivo=None):
    """
    Avalia o modelo e imprime as métricas completas do projeto:
    Precisão (macro), Recall (macro) e F1-macro.

    Parâmetros:
        modelo       : rede neural (com os pesos do melhor modelo carregados)
        loader_val   : DataLoader de validação (com rótulos)
        nomes_classes: lista com os nomes das classes
        dispositivo  : 'cpu' ou 'cuda'

    Retorna:
        dicionário com as métricas principais
    """

    if dispositivo is None:
        dispositivo = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    modelo.eval()
    modelo.to(dispositivo)

    # Listas para guardar todas as predições e rótulos reais
    todas_predicoes = []
    todos_rotulos   = []

    for imagens, rotulos in loader_val:
        imagens = imagens.to(dispositivo)

        # Obtém as predições (classe com maior pontuação)
        predicoes = modelo(imagens).argmax(1).cpu().tolist()
        todas_predicoes.extend(predicoes)
        todos_rotulos.extend(rotulos.tolist())

    # --- Calcula as métricas com sklearn ---

    # Precisão macro: média da precisão de cada classe individualmente
    # "De todas as vezes que predi Cereja, quantas eram realmente Cereja?"
    precisao = precision_score(
        todos_rotulos, todas_predicoes,
        average="macro", zero_division=0
    )

    # Recall macro: média do recall de cada classe individualmente
    # "De todas as Cerejas reais, quantas eu identifiquei corretamente?"
    recall = recall_score(
        todos_rotulos, todas_predicoes,
        average="macro", zero_division=0
    )

    # F1-macro: média harmônica entre Precisão e Recall por classe
    # É a métrica principal da avaliação — equilibra precisão e recall
    f1 = f1_score(
        todos_rotulos, todas_predicoes,
        average="macro", zero_division=0
    )

    # Exibe o relatório completo
    print("\n" + "=" * 55)
    print("RESULTADO DA AVALIAÇÃO")
    print("=" * 55)
    print(f"  Precisão (macro) : {precisao:.4f}")
    print(f"  Recall   (macro) : {recall:.4f}")
    print(f"  F1-macro         : {f1:.4f}  ← métrica principal")
    print("\nDetalhamento por classe:")
    print(classification_report(
        todos_rotulos, todas_predicoes,
        target_names=nomes_classes,
        zero_division=0
    ))
    print("Matriz de Confusão:")
    print("(linhas = classe real | colunas = classe predita)")
    print(confusion_matrix(todos_rotulos, todas_predicoes))
    print("=" * 55)

    return {
        "precisao_macro": precisao,
        "recall_macro"  : recall,
        "f1_macro"      : f1,
        "predicoes"     : todas_predicoes,
        "rotulos"       : todos_rotulos,
    }


