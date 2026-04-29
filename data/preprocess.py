import pandas as pd

def preprocess_data(df):
    """
    Realiza o pré-processamento dos dados
    """

    # Remove coluna ID (não relevante)
    df = df.drop(columns=["id"], errors="ignore")

    # Converte diagnóstico:
    # M (maligno) → 1
    # B (benigno) → 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Separa features (X) e target (y)
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]

    return X, y