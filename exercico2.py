import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris, load_wine 
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Lista com os carregadores de dados
datasets = [
    ("IRIS", load_iris()),
    ("WINE", load_wine())
]

porcentagens = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
valores_k = [1, 3, 5, 7, 9] # Valores ímpares para desempate
repeticoes = 20

# 2. Loop Principal: Um para cada Dataset
for nome, base in datasets:
    X, y = base.data, base.target
    
    print(f"\n{'='*25} DATASET: {nome} {'='*25}")
    print(f"{'Treino %':<10} | {'K':<3} | {'Média':<8} | {'Máx':<8} | {'Mín':<8} | {'Variância':<8}")
    print("-" * 75)

    # Dicionário para guardar as médias de cada K para o gráfico
    # Ex: {1: [medias], 3: [medias]...}
    historico_grafico = {k: [] for k in valores_k}

    for p in porcentagens:
        for k in valores_k:
            acuracias_temporarias = []
            
            for i in range(repeticoes):
                # O train_test_split gera uma divisão aleatória a cada repetição
                X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p)
                
                # Configura o KNN com o K do loop atual
                knn = KNeighborsClassifier(n_neighbors=k)
                knn.fit(X_train, y_train)
                pred = knn.predict(X_test)
                
                acuracias_temporarias.append(accuracy_score(y_test, pred))
            
            # Cálculos Estatísticos
            media = np.mean(acuracias_temporarias)
            maximo = np.max(acuracias_temporarias) 
            minimo = np.min(acuracias_temporarias)
            variancia = np.var(acuracias_temporarias)

            # Guarda a média para o gráfico deste K específico
            historico_grafico[k].append(media)

            # Exibição na tabela
            print(f"{int(p*100):>8}% | {k:<3} | {media:.4f} | {maximo:.4f} | {minimo:.4f} | {variancia:.6f}")
        print("-" * 75)

    # --- GERAÇÃO DO GRÁFICO PARA O DATASET ATUAL ---
    plt.figure(figsize=(10, 6))
    
    for k in valores_k:
        # Plota uma linha para cada valor de K
        plt.plot(porcentagens, historico_grafico[k], marker='o', label=f'K = {k}')

    plt.title(f'Desempenho K-NN: Acurácia Média vs % Treino ({nome})')
    plt.xlabel('Proporção de Treino')
    plt.ylabel('Acurácia Média')
    plt.xticks(porcentagens, [f"{int(x*100)}%" for x in porcentagens])
    plt.legend() # Mostra a legenda identificando qual cor é qual K
    plt.grid(True)
    plt.show() # Abre o gráfico (feche para o código continuar para o próximo dataset)

print("\nExercício 2 concluído.")