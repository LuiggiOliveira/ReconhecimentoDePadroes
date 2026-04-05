import numpy as np
import matplotlib.pyplot as plt # Nova importação: para criar os gráficos
from sklearn.datasets import load_iris # busca as tabelas do dataset iris
from sklearn.model_selection import train_test_split # 'corta' os dados, pegando uma parte para o computador estudar e outra para testar
from sklearn.neighbors import KNeighborsClassifier # implementa a distancia euclidiana
from sklearn.metrics import accuracy_score # compara o que o computador chutou com o que era a resposta real e diz a porcentagem de acerto

# 1. Carrega os dados do dataset iris que já estão dentro do venv após importar as bibliotecas
iris = load_iris()
X, y = iris.data, iris.target

# 2. Configurações do experimento
porcentagens = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8] # varia de 10% a 80% do quanto pegamos da tabela inicial
repeticoes = 20  # quantas vezes vamos rodar para cada %

# Estrutura para o gráfico: lista para guardar as médias
medias_para_grafico = []

print(f"\n{'='*25} DATASET: IRIS (NN: K=1) {'='*25}")
print(f"{'Treino %':<10} | {'Média':<8} | {'Máximo':<8} | {'Mínimo':<8} | {'Variância':<8}")
print("-" * 65)

# 3. Loop principal (cada porcentagem)
for p in porcentagens:
    acuracias_da_rodada = [] # para cada %, eu faço 20 repetições e guardo nesse vetor 
    
    for i in range(repeticoes):
        # Divide os dados (train_size é a porcentagem atual)
        # shuffle=True (padrão) garante que o sorteio seja diferente a cada rodada
        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=p)
        
        # Cria o modelo NN (K=1) conforme exigido pelo exercicio, para buscar o vizinho mais próximo
        modelo = KNeighborsClassifier(n_neighbors=1)
        
        # Treina e testa
        modelo.fit(X_train, y_train) # é aqui onde o computador vai treinar/estudar, olhando para os dados do treino
        previsoes = modelo.predict(X_test) # depois é aqui que vai calculando as distancias euclidianas até onde ele memorizou no fit 
        
        # Calcula a acurácia e guarda na lista temporária das 20 rodadas
        acc = accuracy_score(y_test, previsoes)
        acuracias_da_rodada.append(acc)
    
    # 4. Cálculos Estatísticos usando NumPy sobre as 20 repetições
    media = np.mean(acuracias_da_rodada) 
    maximo = np.max(acuracias_da_rodada)
    minimo = np.min(acuracias_da_rodada)
    variancia = np.var(acuracias_da_rodada)
    
    # Guardamos a média para usar no gráfico no final
    medias_para_grafico.append(media)
    
    # Exibe os resultados formatados na tabela
    print(f"{int(p*100):>8}% | {media:.4f} | {maximo:.4f} | {minimo:.4f} | {variancia:.6f}")

# OBS: quando se têm, por exemplo, 10% no treino, a variância tende a ser maior

# --- PARTE DO GRÁFICO (MATPLOTLIB) ---
print("\nGerando gráfico...")

# Configura a janela do gráfico (largura, altura em polegadas)
plt.figure(figsize=(10, 6))

# Plota a linha: eixo X é a % de treino, eixo Y é a média acumulada
# marker='o' coloca bolinhas nos pontos, linestyle='-' faz a linha contínua, color='b' é azul
plt.plot(porcentagens, medias_para_grafico, marker='o', linestyle='-', color='b')

# Títulos e Rótulos dos Eixos (Dashboard do professor)
plt.title('Acurácia Média vs % de Treino (Classificador NN - Iris Data)')
plt.xlabel('Proporção de Treino (X-axis)')
plt.ylabel('Taxa de Acerto Média (Y-axis)')

# Formata o eixo X para mostrar porcentagens bonitinhas (10%, 20%...)
plt.xticks(porcentagens, [f"{int(x*100)}%" for x in porcentagens])

# Ativa o quadriculado de fundo para facilitar a leitura
plt.grid(True)

# Exibe o gráfico na tela (vai abrir uma janela nova)
plt.show()

print("Exercício 1 concluído.")