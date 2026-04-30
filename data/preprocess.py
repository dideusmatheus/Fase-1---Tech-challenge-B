import pandas as pd

def preprocess_data(df):
    """
    Realiza o pré-processamento dos dados
    """

    # Remove coluna ID (não relevante)
    df = df.drop(columns=["id"], errors="ignore")

    # 🔎 Remove colunas com muitos valores ausentes (NaN)
    # Ex: remove colunas com mais de 40% de valores faltantes
    threshold = 0.4
    df = df.loc[:, df.isnull().mean() < threshold]

    # Converte diagnóstico:
    # M (maligno) → 1
    # B (benigno) → 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Remove linhas onde target ficou nulo após o map (segurança)
    # df = df.dropna(subset=["diagnosis"])

    # Separa features (X) e target (y)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return X, y