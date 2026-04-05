import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

# 1. Carrega os datasets (Igual aos anteriores para manter o padrão)
datasets = [
    ("IRIS", load_iris()),
    ("WINE", load_wine())
]

porcentagens = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
repeticoes = 20

# 2. Loop Principal por Dataset
for nome, base in datasets:
    X, y = base.data, base.target
    classes = np.unique(y) # identifica as classes (ex: 0, 1, 2 no Iris)
    
    print(f"\n{'='*25} DATASET: {nome} (DMC) {'='*25}")
    print(f"{'Treino %':<10} | {'Média':<8} | {'Máx':<8} | {'Mín':<8} | {'Variância':<8}")
    print("-" * 65)

    medias_para_grafico = []
    
    # Variável para guardar a última matriz de confusão (para mostrar no final)
    ultima_matriz = None

    for p in porcentagens:
        acuracias_da_rodada = []
        
        for i in range(repeticoes):
            # Divide os dados (estratégia igual aos outros exercícios)
            X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p)
            
            # --- IMPLEMENTAÇÃO DO DMC (Cálculo dos Centroides) ---
            centroides = []
            for c in classes:
                # Pega apenas as linhas que pertencem à classe 'c' no treino
                pontos_da_classe = X_train[y_train == c]
                
                # Se houver pontos, tira a média (centroide). Se não, usa zeros.
                if len(pontos_da_classe) > 0:
                    centroides.append(np.mean(pontos_da_classe, axis=0))
                else:
                    centroides.append(np.zeros(X.shape[1]))
            
            centroides = np.array(centroides)

            # --- PREDIÇÃO NO TESTE (Distância Euclidiana até o Centroide) ---
            previsoes = []
            for ponto_teste in X_test:
                # Calcula a distância do ponto de teste para cada um dos centroides
                distancias = np.linalg.norm(centroides - ponto_teste, axis=1)
                # O índice do centroide mais perto é a nossa previsão (classe)
                previsoes.append(np.argmin(distancias))
            
            previsoes = np.array(previsoes)
            
            # Calcula a acurácia e guarda para a estatística
            acc = accuracy_score(y_test, previsoes)
            acuracias_da_rodada.append(acc)
            
            # Se for a última repetição do último treino (80%), guarda a matriz
            if p == 0.8 and i == repeticoes - 1:
                ultima_matriz = confusion_matrix(y_test, previsoes)

        # 3. Cálculos Estatísticos
        media = np.mean(acuracias_da_rodada)
        maximo = np.max(acuracias_da_rodada)
        minimo = np.min(acuracias_da_rodada)
        variancia = np.var(acuracias_da_rodada)
        
        medias_para_grafico.append(media)
        
        print(f"{int(p*100):>8}% | {media:.4f} | {maximo:.4f} | {minimo:.4f} | {variancia:.6f}")

    # --- GRÁFICO DE ACURÁCIA MÉDIA ---
    plt.figure(figsize=(8, 5))
    plt.plot(porcentagens, medias_para_grafico, marker='s', color='green', label='DMC')
    plt.title(f'DMC: Taxa Média de Acerto vs % Treino ({nome})')
    plt.xlabel('Proporção de Treino')
    plt.ylabel('Acurácia Média')
    plt.grid(True)
    plt.show()

    # --- APRESENTAÇÃO DA MATRIZ DE CONFUSÃO ---
    # Mostra onde o computador acertou e onde ele 'confundiu' as classes
    disp = ConfusionMatrixDisplay(confusion_matrix=ultima_matriz, display_labels=base.target_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matriz de Confusão Final ({nome})')
    plt.show()

print("\nExercício 3 concluído.")