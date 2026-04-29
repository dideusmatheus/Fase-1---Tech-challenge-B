from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def train_models(X, y):
    """
    Treina modelos de machine learning
    """

    # Divide dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y  # mantém proporção das classes
    )

    # Modelo 1: Regressão Logística
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)

    # Modelo 2: Random Forest
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    return lr, rf, X_test, y_test