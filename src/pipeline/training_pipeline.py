from src.machine_learning.data.load_data import load_data
from src.machine_learning.data.preprocess import preprocess_data
from src.machine_learning.training import run_training
from src.machine_learning.validation import run_validation
from src.machine_learning.test import run_test

def run_pipeline():
    """
    Pipeline completo do projeto de diagnóstico de câncer de mama.

    Fluxo:
      1. Carrega o CSV bruto
      2. Pré-processa e gera splits (treino / validação / teste)
      3. Treina todos os modelos com dados de treino
      4. Valida todos os modelos → escolhe o melhor
      5. Avalia o melhor modelo no teste final
    """

    # ── ETAPA 1: CARGA DOS DADOS ──────────────────────────────
    print("📂 Carregando dados brutos...")
    df = load_data("data/machine_learning/raw/data.csv")

    # ── ETAPA 2: PRÉ-PROCESSAMENTO ────────────────────────────
    # Retorna os 6 splits prontos e já escalonados
    # Também salva arquivos em data/machine_learning/processed/
    X_train, X_val, X_test, y_train, y_val, y_test = preprocess_data(df)

    # ── ETAPA 3: TREINAMENTO ──────────────────────────────────
    # Treina todos os modelos com X_train
    # Avalia accuracy em X_val durante o treino (monitoramento)
    trained_models = run_training(X_train, X_val, y_train, y_val)

    # ── ETAPA 4: VALIDAÇÃO ────────────────────────────────────
    # Compara todos os modelos no conjunto de validação
    # Escolhe o melhor com base no Recall de Maligno
    best_name, best_model = run_validation(trained_models, X_val, y_val)

    # ── ETAPA 5: TESTE FINAL ──────────────────────────────────
    # Avalia o melhor modelo no conjunto de teste (roda apenas uma vez)
    # Gera métricas finais: accuracy, recall, F1, confusion matrix
    metrics = run_test(best_model, best_name, X_test, y_test)

    # Retorna métricas para uso externo (notebooks, relatório)
    return metrics


if __name__ == "__main__":
    run_pipeline()