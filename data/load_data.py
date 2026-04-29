# Importa pandas para manipulação de dados
import pandas as pd

def load_data(path):
    """
    Carrega o dataset a partir de um arquivo CSV
    """

    # Lê o arquivo CSV
    df = pd.read_csv(path)

    # Retorna o DataFrame
    return df


if __name__ == "__main__":
    # Caminho do dataset
    path = "data/raw/data.csv"

    # Carrega os dados
    df = load_data(path)

    # Mostra as primeiras linhas
    print(df.head())

    # Mostra informações gerais
    print(df.info())