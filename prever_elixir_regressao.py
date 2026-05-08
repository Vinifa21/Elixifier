import pandas as pd
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, classification_report

def train_elixir_model_reg(csv_path):
    # 1. Ler o arquivo CSV
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_path}' não encontrado. Por favor, verifique o caminho.")
        return

    # 2. Pré-processamento
    target_col = 'Cost'
    if target_col not in df.columns:
        print(f"Erro: Coluna de custo '{target_col}' não encontrada no CSV.")
        return




    # Removemos a coluna 'Card' pois é apenas um nome
    X = df.drop(columns=[target_col, 'Card'])
    y = df[target_col]



    # 3. Separar em dados de treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 4. Criar e treinar o modelo de Árvore de Decisão Regressora
    # Substituímos DecisionTreeClassifier por DecisionTreeRegressor
    model = DecisionTreeRegressor(random_state=42, max_depth=3)
    model.fit(X_train, y_train)

    # 5. Avaliar o modelo
    y_pred_raw = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred_raw)
    mse = mean_squared_error(y_test, y_pred_raw)
    rmse = np.sqrt(mse)
    
    # Aproximação para o custo inteiro mais adequado usando NumPy (np.round)
    y_pred_rounded = np.round(y_pred_raw).astype(int)
    
    print("=== Avaliação do Modelo ===")
    print(f"Erro Médio Absoluto (MAE): {mae:.2f}")
    print(f"Erro Quadrático Médio (MSE): {mse:.2f}")
    print(f"Raiz do Erro Quadrático Médio (RMSE): {rmse:.2f}\n")
    print(f"Acurácia (após arredondamento): {accuracy_score(y_test, y_pred_rounded):.2f}\n")
    
    print("Relatório de Classificação (após arredondamento):")
    print(classification_report(y_test, y_pred_rounded, zero_division=0))

    feature_importances = pd.DataFrame(
        {'Feature': X.columns, 'Importância': model.feature_importances_}
    ).sort_values(by='Importância', ascending=False)
    
    print("\n=== Importância das Variáveis ===")
    print(feature_importances.head(10))

    return model

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    nome_do_arquivo = os.path.join(script_dir, "clash_royale_cards.csv")
    train_elixir_model_reg(nome_do_arquivo)
