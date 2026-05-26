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
