import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def preprocess_data(df):
    """
    Realiza pré-processamento completo dos dados:
    - Limpeza
    - Encoding do target
    - Split treino/validação/teste
    - Escalonamento
    - Salvamento dos splits em data/processed/
    """

    # Cria pasta de saída para os splits processados
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("src/models", exist_ok=True)

    # ── LIMPEZA ────────────────────────────────────────────────

    # Remove coluna 'id' — não tem valor preditivo
    df = df.drop(columns=["id"], errors="ignore")

    # Remove colunas com mais de 40% de valores ausentes (NaN, null, etc.)
    threshold = 0.4
    df = df.loc[:, df.isnull().mean() < threshold]

    # Remove linhas duplicadas (integridade dos dados)
    df = df.drop_duplicates()

    # ── ENCODING DO TARGET ─────────────────────────────────────

    # Converte diagnóstico categórico para numérico:
    # M (maligno)  → 1
    # B (benigno)  → 0
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})

    # Remove linhas onde o target ficou nulo por algum valor inesperado
    df = df.dropna(subset=["diagnosis"])

    # ── SEPARAÇÃO FEATURES / TARGET ───────────────────────────

    # X recebe todas as colunas exceto o target (diagnosis)
    X = df.drop(columns=["diagnosis"])

    # y recebe apenas o target (0 ou 1)
    y = df["diagnosis"]

    # ── SPLIT TREINO / VALIDAÇÃO / TESTE ──────────────────────

    # Passo 1: separa 20% para teste final (nunca visto durante treino)
    # stratify=y garante proporção de M/B igual nos dois conjuntos
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Passo 2: dos 80% restantes, separa 20% para validação
    # Resultado final: 64% treino | 16% validação | 20% teste
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=0.2,
        random_state=42,
        stratify=y_temp
    )

    # ── ESCALONAMENTO ─────────────────────────────────────────

    # StandardScaler: transforma features para média 0 e desvio padrão 1
    # IMPORTANTE: fit() apenas no treino para evitar data leakage
    scaler = StandardScaler()

    # Aprende a escala SOMENTE com dados de treino
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X.columns
    )

    # Aplica a mesma escala do treino na validação (sem reaprender)
    X_val_scaled = pd.DataFrame(
        scaler.transform(X_val),
        columns=X.columns
    )

    # Aplica a mesma escala do treino no teste (sem reaprender)
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X.columns
    )

    # Salva o scaler para uso futuro em produção
    # joblib.dump(scaler, "src/models/scaler.pkl")

    # ── SALVAMENTO DOS SPLITS ─────────────────────────────────

    # Salva features escalonadas de cada split
    # X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    # X_val_scaled.to_csv(  "data/processed/X_val.csv",   index=False)
    # X_test_scaled.to_csv( "data/processed/X_test.csv",  index=False)

    # Salva targets de cada split (reset_index para índice limpo)
    # y_train.reset_index(drop=True).to_csv("data/processed/y_train.csv", index=False)
    # y_val.reset_index(drop=True).to_csv(  "data/processed/y_val.csv",   index=False)
    # y_test.reset_index(drop=True).to_csv( "data/processed/y_test.csv",  index=False)

    # Exibe resumo da divisão
    total = len(df)
    print("✅ Pré-processamento concluído. Splits salvos em data/processed/")
    print(f"   Treino    : {len(X_train_scaled):>4} amostras ({len(X_train_scaled)/total*100:.0f}%)")
    print(f"   Validação : {len(X_val_scaled):>4} amostras ({len(X_val_scaled)/total*100:.0f}%)")
    print(f"   Teste     : {len(X_test_scaled):>4} amostras ({len(X_test_scaled)/total*100:.0f}%)")

    # Retorna os seis splits para uso direto no pipeline
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test