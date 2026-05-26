import pandas as pd
import joblib
import os
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
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

    # ── RESUMO ──────────────────────────────────────

    # Matriz mostra: TN | FP
    #                FN | TP
    # FN (linha Maligno, coluna Benigno) deve ser o mais baixo possível

    # Detalha cada quadrante para facilitar interpretação no relatório
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    print(f"\n  Verdadeiros Negativos (TN) : {tn}  → Benignos corretamente identificados")
    print(f"  Falsos Positivos     (FP) : {fp}  → Benignos classificados como Malignos")
    print(f"  Falsos Negativos     (FN) : {fn}  → ⚠️ Malignos classificados como Benignos")
    print(f"  Verdadeiros Positivos(TP) : {tp}  → Malignos corretamente identificados")

    # ── IDENTIFICAÇÃO DOS FALSOS NEGATIVOS ────────────────────
    fn_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 0]
    if fn_indices:
        id_path = "data/machine_learning/processed/id_test.csv"
        if os.path.exists(id_path):
            id_test = pd.read_csv(id_path).squeeze()
            fn_ids = [id_test.iloc[i] for i in fn_indices]
            print("\n  ⚠️  Paciente(s) com falso negativo (ID original):")
            for pid in fn_ids:
                print(f"     → ID: {pid}")

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
        "confusion_matrix": confusion_matrix(y_test, y_pred),
    }
