import pandas as pd
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

def run_validation(trained_models, X_val, y_val):
    """
    Responsabilidade: VALIDAR os modelos treinados.
    - Avalia cada modelo no conjunto de validação
    - Compara métricas entre todos os modelos
    - Identifica o melhor modelo com base no Recall de Maligno
    - Retorna o nome e o objeto do melhor modelo
    
    ⚠️ Conjunto de validação serve para ESCOLHER o melhor modelo.
       Nunca use o conjunto de teste nesta etapa.
    """

    print("\n" + "="*55)
    print("🔍 VALIDAÇÃO DOS MODELOS")
    print("    (usado para comparar e escolher o melhor)")
    print("="*55)

    # Dicionário para armazenar o recall de maligno de cada modelo
    # Recall de maligno é a métrica principal: minimizar falsos negativos
    recall_scores = {}

    for name, model in trained_models.items():

        print(f"\n🔹 {name.replace('_', ' ').title()}")
        print("-" * 40)

        # Gera predições no conjunto de validação
        y_pred = model.predict(X_val)

        # ── Accuracy: percentual geral de acertos
        acc = accuracy_score(y_val, y_pred)
        print(f"  Accuracy  : {acc:.4f}")

        # ── AUC-ROC: capacidade de separar as classes (0.5=aleatório, 1=perfeito)
        if hasattr(model, "predict_proba"):
            # predict_proba retorna probabilidade para cada classe
            y_proba = model.predict_proba(X_val)[:, 1]  # pega coluna da classe positiva (Maligno)
            auc = roc_auc_score(y_val, y_proba)
            print(f"  AUC-ROC   : {auc:.4f}")

        # ── Classification Report: precision, recall e F1 por classe
        report = classification_report(
            y_val, y_pred,
            target_names=["Benigno (B)", "Maligno (M)"],
            output_dict=True  # retorna dict para extrair recall de maligno
        )
        print(classification_report(
            y_val, y_pred,
            target_names=["Benigno (B)", "Maligno (M)"]
        ))

        # ── Confusion Matrix: TP, TN, FP, FN
        print("  Confusion Matrix:")
        print(f"  {confusion_matrix(y_val, y_pred)}")

        # Armazena o recall da classe Maligno para comparação final
        # Recall Maligno = TP / (TP + FN) → % de casos malignos detectados
        recall_maligno = report["Maligno (M)"]["recall"]
        recall_scores[name] = recall_maligno

    # ── COMPARATIVO FINAL ─────────────────────────────────────

    print("\n" + "="*55)
    print("📊 COMPARATIVO — RECALL MALIGNO (validação)")
    print("   (quanto maior, menos falsos negativos)")
    print("="*55)

    # Ordena modelos por recall de maligno decrescente
    sorted_scores = sorted(recall_scores.items(), key=lambda x: x[1], reverse=True)

    for rank, (name, recall) in enumerate(sorted_scores, 1):
        # Destaca o melhor modelo com emoji
        star = "🥇" if rank == 1 else f"  {rank}."
        print(f"  {star} {name:<25} Recall Maligno: {recall:.4f}")

    # Identifica o melhor modelo (maior recall de maligno)
    best_name = sorted_scores[0][0]
    best_model = trained_models[best_name]

    print(f"\n✅ Melhor modelo na validação: {best_name}")
    print("   → Será usado na avaliação final com dados de teste.")

    # Retorna o nome e objeto do melhor modelo
    return best_name, best_model


if __name__ == "__main__":
    # Execução direta: carrega splits e modelos do disco
    print("📂 Carregando splits e modelos salvos...")

    # Carrega dados de validação
    X_val = pd.read_csv("data/machine_learning/processed/X_val.csv")
    y_val = pd.read_csv("data/machine_learning/processed/y_val.csv").squeeze()

    # Carrega todos os modelos salvos em src/machine_learning/models/model/
    trained_models = {}
    model_dir = "src/machine_learning/models/model"

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")           # remove extensão
            trained_models[name] = joblib.load(f"{model_dir}/{filename}")

    # Executa validação
    run_validation(trained_models, X_val, y_val)