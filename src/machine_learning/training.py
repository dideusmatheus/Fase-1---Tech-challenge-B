import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import os

def run_training(X_train, X_val, y_train, y_val):
    """
    Responsabilidade: TREINAR os modelos.
    - Recebe os splits de treino e validação já prontos
    - Treina cada modelo com X_train / y_train
    - Avalia accuracy em X_val para monitorar overfitting
    - Salva cada modelo em models/machine_learning
    - Retorna dicionário {nome: modelo_treinado}
    """

    # Cria pasta para salvar os modelos treinados
    os.makedirs("models/machine_learning", exist_ok=True)

    # ── DEFINIÇÃO DOS MODELOS ─────────────────────────────────

    # Dicionário com nome → instância de cada algoritmo
    # Todos com random_state=42 para reprodutibilidade
    models = {
        # Regressão Logística: modelo linear, boa baseline, interpretável
        "logistic_regression": LogisticRegression(max_iter=10000, random_state=42),
        # Random Forest: ensemble de árvores, robusto a outliers
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42),
        # Árvore de Decisão: interpretável, mostra regras de decisão
        "decision_tree": DecisionTreeClassifier(random_state=42),
        # KNN: classifica por similaridade com os k vizinhos mais próximos
        "knn": KNeighborsClassifier(n_neighbors=5),
        # SVM: busca o hiperplano de máxima margem entre classes
        "svm": SVC(probability=True, kernel="rbf", random_state=42),
        # Gradient Boosting: boosting sequencial, alta performance
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        # Extra Trees: similar ao RF mas com splits aleatórios, mais rápido
        "extra_trees": ExtraTreesClassifier(n_estimators=100, random_state=42),
        # MLP: rede neural multicamada, captura padrões não-lineares
        "mlp": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }

    # ── TREINAMENTO ───────────────────────────────────────────

    # Dicionário que irá guardar os modelos já treinados
    trained = {}

    print("\n" + "="*55)
    print("🏋️  TREINAMENTO DOS MODELOS")
    print("="*55)

    for name, model in models.items():

        print(f"\n⏳ Treinando: {name.upper()}")

        # Treina o modelo APENAS com dados de treino
        model.fit(X_train, y_train)

        # Avalia métricas no conjunto de VALIDAÇÃO
        # (serve para comparar modelos e detectar overfitting)
        y_pred_val = model.predict(X_val)

        # Métricas de avaliação no conjunto de validação (monitoramento)
        # O que é: percentual geral de acertos

        # O que é: % de todas as predições que o modelo acertou (benignos + malignos juntos).
        # No seu caso: Métrica enganosa. Um modelo que sempre chuta "benigno" teria ~63% de accuracy
        # (pois ~63% do dataset é benigno), mas erraria todos os malignos. Não use como critério principal.
        val_acc = accuracy_score(y_val, y_pred_val)
        print(f"✅ Concluído | Accuracy Validação:  {val_acc:.4f}")

        # O que é: Dos pacientes que o modelo disse "maligno", quantos % realmente eram malignos.
        # No seu caso: Mede falsos alarmes — quantas biopsias desnecessárias o modelo gera. Secundário.
        precision = precision_score(y_val, y_pred_val)
        print(f"✅ Concluído | Precision Validação: {precision:.4f}")

        # O que é: Dos pacientes que realmente têm câncer, quantos % o modelo detectou corretamente.
        # No seu caso: MÉTRICA PRINCIPAL. Recall baixo = falsos negativos = pacientes enviados pra casa sem tratamento.
        recall = recall_score(y_val, y_pred_val)
        print(f"✅ Concluído | Recall Validação:    {recall:.4f}")

        # O que é: Média harmônica entre Precision e Recall. Penaliza quando um dos dois é muito baixo.
        # No seu caso: Visão balanceada, mas como você prioriza Recall, use como métrica de apoio.
        f1 = f1_score(y_val, y_pred_val)
        print(f"✅ Concluído | F1 Score Validação:  {f1:.4f}")

        # Salva o modelo treinado em disco (.pkl)
        joblib.dump(model, f"models/machine_learning/{name}.pkl")

        # Armazena o modelo no dicionário de retorno
        trained[name] = model

    print("\n" + "="*55)
    print(f"💾 {len(trained)} modelos salvos em models/machine_learning/")
    print("="*55)

    # Retorna todos os modelos treinados
    return trained
