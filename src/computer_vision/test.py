import numpy as np          # operações numéricas com arrays
import tensorflow as tf      # framework de deep learning (para iterar o dataset)
from sklearn.metrics import (
    classification_report,   # relatório de precisão, recall e F1 por classe
    confusion_matrix,         # matriz de acertos e erros
    roc_auc_score,            # área sob a curva ROC
    accuracy_score            # percentual de acertos geral
)


def run_test_vision(modelo, test_ds):
    """
    Responsabilidade: AVALIAÇÃO FINAL da CNN no conjunto de teste.
    - Roda UMA única vez com o modelo escolhido após a validação
    - Gera as métricas definitivas: accuracy, AUC, recall, F1
    - Exibe interpretação clínica dos resultados

    ⚠️ REGRA DE OURO: o conjunto de teste NUNCA deve ser visto
       antes desta etapa. Qualquer ajuste de modelo ou threshold
       feito APÓS ver os resultados de teste invalida a avaliação
       (data leakage no processo de avaliação).
    """

    print("\n" + "="*55)
    print("🏁 AVALIAÇÃO FINAL DA CNN — CONJUNTO DE TESTE")
    print("="*55)

    # ── COLETA PREDIÇÕES ──────────────────────────────────────

    # Listas para acumular probabilidades e rótulos de todos os batches
    todas_probs  = []  # probabilidades de ser MALIGNO (saída do sigmoid)
    todos_labels = []  # rótulos reais das imagens de teste

    # Itera batch a batch pelo dataset de teste
    for imagens, labels in test_ds:

        # Gera a probabilidade de ser MALIGNO para cada imagem do batch
        # verbose=0: sem barra de progresso individual por batch
        probs = modelo.predict(imagens, verbose=0)

        # Achata a saída de (batch, 1) para (batch,) e adiciona à lista
        todas_probs.extend(probs.flatten().tolist())

        # Converte tensor de labels para lista Python e adiciona
        todos_labels.extend(labels.numpy().tolist())

    # Converte listas Python para arrays NumPy para uso com sklearn
    todas_probs  = np.array(todas_probs)   # probabilidades contínuas [0.0, 1.0]
    todos_labels = np.array(todos_labels)  # rótulos binários {0, 1}

    # ── BINARIZA PREDIÇÕES ────────────────────────────────────

    # Aplica threshold 0.5: probabilidade >= 0.5 → MALIGNO (1), senão BENIGNO (0)
    preds_binarias = (todas_probs >= 0.5).astype(int)

    # ── ACCURACY ──────────────────────────────────────────────

    # Percentual geral de acertos entre todas as predições
    acc = accuracy_score(todos_labels, preds_binarias)
    print(f"\n  Accuracy  : {acc:.4f}")

    # ── AUC-ROC ───────────────────────────────────────────────

    # Capacidade do modelo de separar casos malignos de benignos
    # Usa as probabilidades brutas (não as classes binárias) → mais informativo
    # Um AUC de 0.95 significa que o modelo ranqueia um maligno acima de um benigno
    # em 95% dos pares comparados
    auc = roc_auc_score(todos_labels, todas_probs)
    print(f"  AUC-ROC   : {auc:.4f}")

    # ── CLASSIFICATION REPORT ─────────────────────────────────

    # Exibe precision, recall e F1 para cada classe individualmente
    # Linha "Maligno (M)": RECALL é a métrica mais crítica
    # Recall Maligno = TP / (TP + FN) = % de cânceres reais que o modelo detectou
    print("\n  Classification Report:")
    print(classification_report(
        todos_labels, preds_binarias,
        target_names=["Benigno (B)", "Maligno (M)"]
    ))

    # ── MATRIZ DE CONFUSÃO DETALHADA ──────────────────────────

    # Decompõe as predições em 4 quadrantes com interpretação clínica
    tn, fp, fn, tp = confusion_matrix(todos_labels, preds_binarias).ravel()

    print(f"\n  Verdadeiros Negativos (TN) : {tn:<4} → Benignos corretamente identificados")
    print(f"  Falsos Positivos     (FP) : {fp:<4} → Benignos classificados como Malignos (alarme falso)")
    print(f"  Falsos Negativos     (FN) : {fn:<4} → ⚠️  Malignos classificados como Benignos (câncer perdido)")
    print(f"  Verdadeiros Positivos(TP) : {tp:<4} → Malignos corretamente identificados")

    # ── EXTRAI MÉTRICAS PRINCIPAIS ────────────────────────────

    # Obtém métricas da classe MALIGNO via classification_report com output_dict
    report_dict    = classification_report(todos_labels, preds_binarias, output_dict=True)
    recall_maligno = report_dict["1"]["recall"]     # % de cânceres detectados
    f1_maligno     = report_dict["1"]["f1-score"]   # equilíbrio entre recall e precision

    # ── RESUMO FINAL ─────────────────────────────────────────

    print("\n" + "="*55)
    print("📋 RESUMO FINAL — CNN (Visão Computacional)")
    print("="*55)
    print(f"  Accuracy        : {acc:.4f}")
    print(f"  AUC-ROC         : {auc:.4f}")
    print(f"  Recall Maligno  : {recall_maligno:.4f}  ← métrica principal")
    print(f"  F1 Maligno      : {f1_maligno:.4f}")
    print(f"  Falsos Negativos: {fn}  ← mamografias malignas não detectadas")
    print("\n  ⚠️  Lembrete: o médico deve ter a palavra final.")
    print("     Este modelo é suporte ao diagnóstico, não um substituto.")
    print("="*55)

    # Retorna dicionário com métricas para uso externo (pipeline, relatório, notebook)
    return {
        "accuracy":           acc,
        "auc":                auc,
        "recall_maligno":     recall_maligno,
        "f1_maligno":         f1_maligno,
        "falsos_negativos":   int(fn),
        "confusion_matrix":   confusion_matrix(todos_labels, preds_binarias),
    }
