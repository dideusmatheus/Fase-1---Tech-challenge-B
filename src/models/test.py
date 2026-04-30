import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    ConfusionMatrixDisplay
)

def run_test(best_model, best_name, X_test, y_test):
    """
    Responsabilidade: AVALIAÇÃO FINAL no conjunto de teste.
    - Roda UMA única vez com o melhor modelo escolhido na validação
    - Gera todas as métricas exigidas pelo projeto (accuracy, recall, F1)
    - Produz o resultado honesto e imparcial do modelo

    ⚠️ REGRA DE OURO: o conjunto de teste NUNCA deve ser visto
       antes desta etapa. Qualquer ajuste de modelo após ver o teste
       invalida a avaliação.
    """

    print("\n" + "="*55)
    print("🏁 AVALIAÇÃO FINAL — CONJUNTO DE TESTE")
    print(f"   Modelo: {best_name}")
    print("="*55)

    # ── PREDIÇÕES ─────────────────────────────────────────────

    # Gera as predições binárias (0=Benigno, 1=Maligno)
    y_pred = best_model.predict(X_test)

    # ── ACCURACY ──────────────────────────────────────────────

    # Percentual geral de acertos (não é a métrica principal aqui)
    acc = accuracy_score(y_test, y_pred)
    print(f"\n  Accuracy  : {acc:.4f}")

    # ── AUC-ROC ───────────────────────────────────────────────

    # Mede a capacidade do modelo separar classes
    # 0.5 = aleatório | 1.0 = perfeito
    if hasattr(best_model, "predict_proba"):
        y_proba = best_model.predict_proba(X_test)[:, 1]  # probabilidade da classe Maligno
        auc = roc_auc_score(y_test, y_proba)
        print(f"  AUC-ROC   : {auc:.4f}")

    # ── CLASSIFICATION REPORT ─────────────────────────────────

    # Exibe precision, recall e F1 para cada classe
    # MÉTRICA PRINCIPAL: Recall de Maligno
    # → Recall = TP / (TP + FN)
    # → Falso Negativo (dizer benigno quando é maligno) = risco de vida
    # → Por isso maximizamos o Recall de Maligno, não apenas a Accuracy
    print("\n  Classification Report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["Benigno (B)", "Maligno (M)"]
    ))

    # ── CONFUSION MATRIX ──────────────────────────────────────

    # Matriz mostra: TN | FP
    #                FN | TP
    # FN (linha Maligno, coluna Benigno) deve ser o mais baixo possível
    cm = confusion_matrix(y_test, y_pred)
    print("  Confusion Matrix:")
    print(f"  {cm}")

    # Detalha cada quadrante para facilitar interpretação no relatório
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Verdadeiros Negativos (TN) : {tn}  → Benignos corretamente identificados")
    print(f"  Falsos Positivos     (FP) : {fp}  → Benignos classificados como Malignos")
    print(f"  Falsos Negativos     (FN) : {fn}  → ⚠️ Malignos classificados como Benignos")
    print(f"  Verdadeiros Positivos(TP) : {tp}  → Malignos corretamente identificados")

    # ── CONCLUSÃO ─────────────────────────────────────────────

    # Extrai recall de maligno para a conclusão
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    LABEL_MALIGNO = "1"
    recall_maligno = report_dict[LABEL_MALIGNO]["recall"]
    f1_maligno     = report_dict[LABEL_MALIGNO]["f1-score"]

    print("\n" + "="*55)
    print("📋 RESUMO FINAL")
    print("="*55)
    print(f"  Modelo          : {best_name}")
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  Recall Maligno  : {recall_maligno:.4f}  ← métrica principal")
    print(f"  F1 Maligno      : {f1_maligno:.4f}")
    print(f"  Falsos Negativos: {fn}  ← casos malignos não detectados")
    print("\n  ⚠️  Lembrete: o médico deve ter a palavra final.")
    print("     Este modelo é um suporte ao diagnóstico, não um substituto.")
    print("="*55)

    # Retorna métricas para uso externo (ex: relatório, notebook)
    return {
        "accuracy":       acc,
        "recall_maligno": recall_maligno,
        "f1_maligno":     f1_maligno,
        "falsos_negativos": fn,
        "confusion_matrix": cm,
    }


if __name__ == "__main__":
    # Execução direta: carrega o melhor modelo e dados de teste do disco

    print("📂 Carregando dados de teste e melhor modelo...")

    # Carrega dados de teste
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").squeeze()

    # Carrega todos os modelos e seleciona o melhor manualmente
    # (em produção, este nome viria do resultado de validation.py)
    model_dir = "src/models/model"
    trained_models = {}

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")
            trained_models[name] = joblib.load(f"{model_dir}/{filename}")

    # Define qual é o melhor modelo (altere conforme resultado da validação)
    best_name  = "random_forest"
    best_model = trained_models[best_name]

    # Executa avaliação final
    run_test(best_model, best_name, X_test, y_test)