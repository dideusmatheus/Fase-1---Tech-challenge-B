import numpy as np          # operações numéricas com arrays
import tensorflow as tf      # framework de deep learning (para iterar o dataset)
from sklearn.metrics import (
    classification_report,   # relatório completo de precisão, recall e F1 por classe
    confusion_matrix,         # matriz de acertos e erros
    roc_auc_score,            # área sob a curva ROC
    accuracy_score            # percentual de acertos geral
)


def coletar_predicoes(modelo, dataset):
    """
    Itera sobre todos os batches do dataset e acumula:
    - probabilidades preditas (saída do sigmoid: float entre 0 e 1)
    - rótulos reais (0 ou 1)

    Necessário porque os dados chegam em batches, não de uma vez só.
    """

    # Listas para acumular as saídas de todos os batches
    todas_probs  = []  # probabilidades de ser MALIGNO (0.0 a 1.0)
    todos_labels = []  # rótulos reais do dataset (0 ou 1)

    # Itera batch a batch pelo dataset de validação
    for imagens, labels in dataset:

        # Gera probabilidades de ser MALIGNO para cada imagem do batch
        # verbose=0: sem barra de progresso por batch (saída fica limpa)
        probs = modelo.predict(imagens, verbose=0)

        # flatten(): transforma a saída de shape (batch, 1) em array 1D (batch,)
        todas_probs.extend(probs.flatten().tolist())

        # numpy(): converte tensor TensorFlow para array NumPy
        todos_labels.extend(labels.numpy().tolist())

    # Converte listas Python para arrays NumPy para uso com sklearn
    todas_probs  = np.array(todas_probs)   # probabilidades contínuas [0.0, 1.0]
    todos_labels = np.array(todos_labels)  # rótulos binários {0, 1}

    return todas_probs, todos_labels


def run_validation_vision(modelo, val_ds):
    """
    Responsabilidade: VALIDAR a CNN no conjunto de validação.
    - Coleta predições em todos os batches
    - Calcula métricas: accuracy, AUC-ROC, recall, F1
    - Exibe classification report e matriz de confusão
    - Retorna dicionário de métricas para uso externo

    ⚠️ Conjunto de validação serve para MONITORAR o treino.
       Nunca use o conjunto de teste nesta etapa.
    """

    print("\n" + "="*55)
    print("🔍 VALIDAÇÃO DA CNN")
    print("    (avaliação no conjunto de validação)")
    print("="*55)

    # ── COLETA PREDIÇÕES ──────────────────────────────────────

    # Acumula probabilidades preditas e rótulos reais de todos os batches
    probs, labels = coletar_predicoes(modelo, val_ds)

    # ── APLICA THRESHOLD ──────────────────────────────────────

    # Converte probabilidades em classes binárias usando threshold 0.5
    # prob >= 0.5 → MALIGNO (1) | prob < 0.5 → BENIGNO (0)
    # Para uso clínico, um threshold menor (ex: 0.3) aumentaria o recall
    # à custa de mais falsos positivos (menos canceres perdidos, mais alarmes)
    threshold = 0.5
    preds_binarias = (probs >= threshold).astype(int)

    # ── ACCURACY ──────────────────────────────────────────────

    # Percentual geral de acertos (benignos + malignos juntos)
    # Atenção: métrica enganosa em datasets desbalanceados
    acc = accuracy_score(labels, preds_binarias)
    print(f"\n  Accuracy  : {acc:.4f}")

    # ── AUC-ROC ───────────────────────────────────────────────

    # Área Sob a Curva ROC: mede a capacidade de separar as duas classes
    # 0.5 = aleatório | 1.0 = separação perfeita | > 0.9 = excelente
    # Usa as probabilidades brutas (não as classes binárias) para ser mais informativo
    auc = roc_auc_score(labels, probs)
    print(f"  AUC-ROC   : {auc:.4f}")

    # ── CLASSIFICATION REPORT ─────────────────────────────────

    # Exibe precision, recall e F1 por classe (Benigno e Maligno)
    # MÉTRICA PRINCIPAL: Recall de Maligno
    # → Recall Maligno baixo = falsos negativos = cânceres não detectados = risco de vida
    print("\n  Classification Report:")
    print(classification_report(
        labels, preds_binarias,
        target_names=["Benigno (B)", "Maligno (M)"]  # nomes legíveis para cada classe
    ))

    # ── MATRIZ DE CONFUSÃO ────────────────────────────────────

    # Decompõe os acertos e erros em 4 categorias:
    # TN: benigno previsto como benigno  (acerto)
    # FP: benigno previsto como maligno  (alarme falso — biopsia desnecessária)
    # FN: maligno previsto como benigno  (PIOR CASO — câncer não detectado)
    # TP: maligno previsto como maligno  (acerto)
    tn, fp, fn, tp = confusion_matrix(labels, preds_binarias).ravel()

    print("  Tabela de Acertos e Erros por Diagnóstico:")
    print("                    Previsto Benigno  -  Previsto Maligno")
    print(f"  Real Benigno      Acerto Benigno = {tn:<5}-  Alarme Falso = {fp}")
    print(f"  Real Maligno   🚨 Câncer Perdido = {fn:<5}-  Acerto Maligno = {tp}")

    # ── EXTRAI MÉTRICAS DA CLASSE MALIGNO ─────────────────────

    # classification_report com output_dict=True retorna dicionário
    report_dict = classification_report(labels, preds_binarias, output_dict=True)

    # A classe MALIGNO tem key "1" no dicionário (pois seu label é 1)
    recall_maligno    = report_dict["1"]["recall"]     # % de malignos detectados
    precision_maligno = report_dict["1"]["precision"]  # % das previsões malignas que eram corretas
    f1_maligno        = report_dict["1"]["f1-score"]   # média harmônica entre recall e precision

    print(f"\n  Recall Maligno    : {recall_maligno:.4f}  ← métrica principal")
    print(f"  Precision Maligno : {precision_maligno:.4f}")
    print(f"  F1 Maligno        : {f1_maligno:.4f}")

    # ── RETORNO ───────────────────────────────────────────────

    # Dicionário com todas as métricas para uso externo (pipeline, relatório)
    return {
        "accuracy":           acc,
        "auc":                auc,
        "recall_maligno":     recall_maligno,
        "precision_maligno":  precision_maligno,
        "f1_maligno":         f1_maligno,
        "falsos_negativos":   int(fn),
        "confusion_matrix":   confusion_matrix(labels, preds_binarias),
    }
