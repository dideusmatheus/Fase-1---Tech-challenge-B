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

    # ── COMPARATIVO FINAL ─────────────────────────────────────

    print("\n" + "="*55)
    print("📊 COMPARATIVO — RECALL MALIGNO (validação)")
    print("   (quanto maior, menos falsos negativos)")
    print("="*55)

    # Ordena modelos por recall de maligno decrescente
    sorted_scores = sorted(recall_scores.items(), key=lambda x: x[1], reverse=True)

    # Imprime o ranking.
    for rank, (name, recall) in enumerate(sorted_scores, 1):
        # Destaca o melhor modelo com emoji
        star = "🥇" if rank == 1 else f"  {rank}."
        print(f"  {star} {name:<25} Recall Maligno: {recall:.4f}")

    # Identifica o melhor modelo (maior recall de maligno)
    # em vez de escolher o modelo mais "preciso" ou com maior accuracy, o projeto escolhe o que menos deixa malignos escaparem, porque no diagnóstico de câncer, um falso negativo tem consequência muito mais grave do que um falso positivo.
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