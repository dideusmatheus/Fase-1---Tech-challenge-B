from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import joblib
import os

def train_models(X, y):
    os.makedirs("src/models", exist_ok=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "src/models/scaler.pkl")

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = {
        "logistic_regression": LogisticRegression(max_iter=10000),
        "random_forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "decision_tree":       DecisionTreeClassifier(random_state=42),
        "knn":                 KNeighborsClassifier(n_neighbors=5),
        "svm":                 SVC(probability=True, kernel="rbf", random_state=42),
        "gradient_boosting":   GradientBoostingClassifier(random_state=42),
        "extra_trees":         ExtraTreesClassifier(n_estimators=100, random_state=42),
        "mlp":                 MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=1000, random_state=42),
    }

    trained = {}
    for name, model in models.items():
        print(f"⏳ Treinando {name}...")
        model.fit(X_train, y_train)
        # joblib.dump(model, f"src/models/model/{name}.pkl") // salva cada modelo individualmente em arquivo separado
        trained[name] = model
        print(f"✅ {name} salvo!")

    # Retorna dict + dados de teste
    return trained, X_test, y_test