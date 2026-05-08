import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os

def agrupar_elixir(custo):
    if custo <= 2:
        return '1-2'
    elif custo <= 4:
        return '3-4'
    elif custo <= 6:
        return '5-6'
    else:
        return '7+'

def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, "clash_royale_cards.csv")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Erro: Arquivo '{csv_path}' não encontrado. Verifique o caminho.")
        return
    # Coluna que se quer prever
    target_col = 'Cost'



    # Separar os recursos (X) e o alvo (y)
    # Selecionar as colunas numéricas de características da carta
    features = ['HP', 'DPS', 'Range', 'Count', 'Has_SplashDamage', 'Is_Air']
    X = df[features]
    
    # Aplicar o agrupamento na coluna alvo
    y = df[target_col].apply(agrupar_elixir)

    # Criar e treinar o modelo com 100% dos dados
    model = DecisionTreeClassifier(random_state=42, max_depth=5)
    model.fit(X, y)

    print("==================================================")
    print("  Modelo treinado com 100% dos dados com sucesso! ")
    print("==================================================\n")
    print("Agora você pode inserir as estatísticas de uma carta fictícia")
    print("e o modelo dirá em qual grupo de elixir ela se encaixa.\n")

    while True:
        try:
            print("-" * 40)
            hp = float(input("HP (Vida da carta, ex: 1000): "))
            dps = float(input("DPS (Dano por segundo, ex: 150): "))
            raio = float(input("Range (Alcance, ex: 5.5 para longo, 1.2 para corpo-a-corpo): "))
            count = float(input("Count (Quantidade de unidades geradas, ex: 1, 3, 15): "))
            splash = float(input("Has_SplashDamage (Dano em área? 1 para Sim, 0 para Não): "))
            is_air = float(input("Is_Air (É voadora? 1 para Sim, 0 para Não): "))
            
            # Criar um DataFrame com a mesma estrutura usada no treinamento
            nova_carta = pd.DataFrame([[hp, dps, raio, count, splash, is_air]], columns=features)
            
            # Fazer a previsão
            previsao = model.predict(nova_carta)[0]
            
            print("\n" + "=" * 40)
            print(f" ESTIMATIVA DE CUSTO: Classe {previsao} de Elixir")
            print("=" * 40 + "\n")
            
            continuar = input("Deseja testar outra carta? (s/n): ")
            if continuar.strip().lower() != 's':
                print("\nEncerrando o programa. Até mais!")
                break
                
        except KeyboardInterrupt:
            print("\nSaindo do programa...")
            break
        except ValueError:
            print("\n Erro: Por favor, digite apenas números nos campos de atributo!\n")

if __name__ == "__main__":
    main()
