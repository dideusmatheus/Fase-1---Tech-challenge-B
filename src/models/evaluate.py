from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(model, X_test, y_test):
    """
    Avalia modelo treinado
    """

    # Faz previsões
    y_pred = model.predict(X_test)

    # Mostra métricas
    print(classification_report(y_test, y_pred))

    # Matriz de confusão
    print(confusion_matrix(y_test, y_pred))