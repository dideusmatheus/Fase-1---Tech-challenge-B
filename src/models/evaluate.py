from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

def evaluate_model(model, X_test, y_test):
    """
    Avalia modelo com métricas adequadas para diagnóstico médico.
    Foco em Recall da classe Maligno (evitar falsos negativos).
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"  Accuracy : {acc:.4f}")

    # AUC-ROC (se modelo suporta predict_proba)
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_proba)
        print(f"  AUC-ROC  : {auc:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
          target_names=["Benigno (B)", "Maligno (M)"]))

    print("  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print()