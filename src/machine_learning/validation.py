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

    # Dicionário para armazenar a Precision Maligno de cada modelo
    # Usado como 2º critério de desempate quando dois modelos têm o mesmo Recall:
    # o que tem maior Precision gera menos alarmes falsos (benignos classificados como malignos)
    precision_scores = {}

    # Dicionário para armazenar o F1 Maligno de cada modelo
    # Usado como 3º critério de desempate quando Recall e Precision também empatam:
    # F1 = média harmônica entre Precision e Recall — equilibra os dois ao mesmo tempo
    f1_scores = {}

    for name, model in trained_models.items():

        print(f"\n🔹 {name.replace('_', ' ').title()}")
        print("-" * 40)

        # Gera predições no conjunto de validação
        y_pred = model.predict(X_val)

        # ── Accuracy: percentual geral de acertos
        # O accuracy_score é exibido separado apenas por ser uma métrica geral rápida
        acc = accuracy_score(y_val, y_pred)
        print(f"  Accuracy  : {acc:.4f}")

        # ── AUC-ROC: capacidade de separar as classes (0.5=aleatório, 1=perfeito)
        # O que é: Mede a capacidade do modelo de separar casos malignos de benignos, independente do threshold de decisão.
        # Funciona assim: para cada threshold possível (0.0 a 1.0), o modelo gera um par (taxa de falsos positivos, taxa de verdadeiros positivos). A curva ROC plota todos esses pares. A AUC é a área sob essa curva.
        # No seu caso: Um modelo com AUC 0.99 consegue quase sempre rankear um maligno com probabilidade maior que um benigno. É uma visão mais robusta que a accuracy.
        if hasattr(model, "predict_proba"):
            # predict_proba retorna probabilidade para cada classe
            y_proba = model.predict_proba(X_val)[:, 1]  # pega coluna da classe positiva (Maligno)
            auc = roc_auc_score(y_val, y_proba)
            print(f"  AUC-ROC   : {auc:.4f}")

        # ── Classification Report: precision, recall e F1 por classe
        # Precision: dos que o modelo disse "Maligno", quantos % realmente eram
        # Recall: dos que realmente eram "Maligno", quantos % o modelo pegou
        # F1: média harmônica entre precision e recall
        # Support: quantidade real de casos naquela classe no conjunto
        # No seu caso: O que importa é a linha Maligno (M) — especialmente o Recall, pois um Recall baixo = falsos negativos = câncer não detectado.
        report = classification_report(
            y_val, y_pred,
            target_names=["Benigno (B)", "Maligno (M)"],
            output_dict=True  # retorna dict para extrair recall de maligno
        )
        print(classification_report(
            y_val, y_pred,
            target_names=["Benigno (B)", "Maligno (M)"]
        ))

        # ── Tabela de Acertos e Erros por Diagnóstico
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
        print("  Tabela de Acertos e Erros por Diagnóstico:")
        print("                    Previsto Benigno  -  Previsto Maligno")
        print(f"  Real Benigno      Acerto Benigno = {tn:<5}-  Alarme Falso = {fp}")
        print(f"  Real Maligno   🚨 Câncer Perdido = {fn:<5}-  Acerto Maligno = {tp}")

        # Armazena o recall da classe Maligno para comparação final
        # Recall Maligno = TP / (TP + FN) → % de casos malignos detectados
        # O classification_report com output_dict=True retorna um dicionário. Esse código extrai especificamente o Recall da classe Maligno de cada modelo e guarda num dicionário recall_scores para comparação posterior.
        recall_maligno = report["Maligno (M)"]["recall"]
        recall_scores[name] = recall_maligno

        precision_maligno = report["Maligno (M)"]["precision"]  # % dos previstos malignos que realmente eram malignos
        precision_scores[name] = precision_maligno              # guarda para 2º desempate no ranking final

        f1_maligno = report["Maligno (M)"]["f1-score"]          # média harmônica entre Precision e Recall
        f1_scores[name] = f1_maligno                            # guarda para 3º desempate no ranking final

    # ── COMPARATIVO FINAL ─────────────────────────────────────

    print("\n" + "="*55)
    print("📊 COMPARATIVO — RECALL MALIGNO (validação)")
    print("   (critério 1: Recall — critério 2: Precision — critério 3: F1)")
    print("="*55)

    # Ordena modelos por três critérios em sequência:
    # 1º Recall Maligno decrescente — minimizar falsos negativos é a prioridade
    # 2º Precision Maligno decrescente — entre recalls iguais, vence quem gera menos alarmes falsos
    # 3º F1 Maligno decrescente — entre recalls e precisions iguais, vence o melhor equilíbrio geral
    sorted_scores = sorted(
        recall_scores.items(),
        key=lambda x: (x[1], precision_scores[x[0]], f1_scores[x[0]]),  # Python compara tupla elemento a elemento
        reverse=True                                                       # decrescente nos três critérios
    )

    # Imprime o ranking com os três critérios lado a lado
    for rank, (name, recall) in enumerate(sorted_scores, 1):
        precision = precision_scores[name]   # recupera a Precision armazenada durante o loop
        f1 = f1_scores[name]                 # recupera o F1 armazenado durante o loop
        star = "🥇" if rank == 1 else f"  {rank}."
        print(f"  {star} {name:<25} Recall: {recall:.4f}  |  Precision: {precision:.4f}  |  F1: {f1:.4f}")

    # Identifica o melhor modelo: maior Recall Maligno, desempatado por Precision Maligno
    # Recall Maligno é a prioridade porque falso negativo (câncer não detectado) tem
    # consequência muito mais grave do que falso positivo (alarme falso)
    best_name = sorted_scores[0][0]          # primeiro da lista ordenada = melhor nos dois critérios
    best_model = trained_models[best_name]

    print(f"\n✅ Melhor modelo na validação: {best_name}")
    print("   → Será usado na avaliação final com dados de teste.")

    # Retorna o nome e objeto do melhor modelo
    return best_name, best_model


if __name__ == "__main__":
    # Execução direta: carrega splits e modelos do disco
    print("📂 Carregando splits e modelos salvos...")

    # Carrega dados de validação
    X_val = pd.read_csv("data/processed/X_val.csv")
    y_val = pd.read_csv("data/processed/y_val.csv").squeeze()

    # Carrega todos os modelos salvos em src/models/model/
    trained_models = {}
    model_dir = "src/models/model"

    for filename in os.listdir(model_dir):
        if filename.endswith(".pkl"):
            name = filename.replace(".pkl", "")           # remove extensão
            trained_models[name] = joblib.load(f"{model_dir}/{filename}")

    # Executa validação
    run_validation(trained_models, X_val, y_val)