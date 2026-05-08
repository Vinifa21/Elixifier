import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

def agrupar_elixir(custo):
    if custo <= 2:
        return '1-2'
    elif custo <= 4:
        return '3-4'
    elif custo <= 6:
        return '5-6'
    else:
        return '7+'

def train_elixir_model_agrupado(csv_path):
    # 1. Ler o arquivo CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_path}' não encontrado. Por favor, verifique o caminho.")
        return

    # 2. Pré-processamento
    target_col = 'Cost'


    # Separar variáveis independentes (X) e a dependente (y)
    # Removemos a coluna 'Card' porque é apenas o nome da carta
    X = df.drop(columns=[target_col, 'Card'])
    
    # Aplicar a função de agrupamento no alvo (y)
    y = df[target_col].apply(agrupar_elixir)



    # 3. Separar em dados de treino e teste
    # Usando test_size=0.2
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # 4. Criar e treinar o modelo de Árvore de Decisão
    # altura máxima igual a 5
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X_train, y_train)

    # 5. Avaliar o modelo
    y_pred = model.predict(X_test)
    
    print("=== Avaliação do Modelo ===")
    print(f"Acurácia: {accuracy_score(y_test, y_pred):.2f}\n")
    print("Relatório de Classificação:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    # Mostrar a importância das features
    feature_importances = pd.DataFrame(
        {'Feature': X.columns, 'Importância': model.feature_importances_}
    ).sort_values(by='Importância', ascending=False)
    
    print("\n=== Importância das Variáveis ===")
    print(feature_importances.head(10))

    print("\n=== Regras da Árvore de Decisão ===")
    regras = export_text(model, feature_names=list(X.columns))
    print(regras)

    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_do_arquivo = os.path.join(script_dir, "clash_royale_cards.csv")
    train_elixir_model_agrupado(nome_do_arquivo)
