#PRIMEIRO MODELO DE TREINAMENTO
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder


def train_elixir_model(csv_path):
    #Tenta ler o arquivo .csv
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_path}' não encontrado. Por favor, verifique o caminho.")
        return

    # Pré-processamento:
    # Aqui sabemos que a coluna alvo é 'Cost'
    target_col = 'Cost'
 


    # Separar variáveis independentes (X) e a dependente (y)
    # Removemos a coluna 'Card' porque é apenas o nome da carta e não influencia seu custo
    X = df.drop(columns=[target_col, 'Card'])
    y = df[target_col]

    


    # Separar em dados de treino e teste
    # 80% para treino, 20% para teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Criar e treinar o modelo de Árvore de Decisão
    # O custo de elixir é um valor discreto (1 a 9)
    model = DecisionTreeClassifier(random_state=42, max_depth=4)
    model.fit(X_train, y_train)

    # Feedback sobre o modelo:
    y_pred = model.predict(X_test)
    
    print("=== Avaliação do Modelo ===")
    #Acurácia = total de acertos / total de testes
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
    print("Relatório de Classificação:")
    #Detalhes sobre o desempenho ao tentar identificar cada classe
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Mostrar a importância de cada coluna (através do Imporeza de Gini)
    feature_importances = pd.DataFrame(
        {'Feature': X.columns, 'Importância': model.feature_importances_}
    ).sort_values(by='Importância', ascending=False)
    
    print("\n=== Importância das Variáveis ===")
    print(feature_importances.head(10))

    # Mostra as regras da árvore
    print("\n=== Regras da Árvore de Decisão ===")
    regras = export_text(model, feature_names=list(X.columns))
    print(regras)

    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_do_arquivo = os.path.join(script_dir, "clash_royale_cards.csv")
    train_elixir_model(nome_do_arquivo)
