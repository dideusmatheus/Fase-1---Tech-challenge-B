from data.load_data import load_data
from data.preprocess import preprocess_data
from src.models.train import train_models
from src.models.evaluate import evaluate_model

def run_pipeline():
    """
    Executa pipeline completo
    """

    # Carrega dados
    df = load_data("data/raw/data.csv")

    # Pré-processa
    X, y = preprocess_data(df)

    # Treina todos os modelos → retorna dict
    trained_models, X_test, y_test = train_models(X, y)

    # Avalia cada modelo automaticamente
    print("\n" + "="*50)
    print("📊 AVALIAÇÃO DOS MODELOS")
    print("="*50)

    for name, model in trained_models.items():
        print(f"\n🔹 {name.replace('_', ' ').title()}:")
        evaluate_model(model, X_test, y_test)

if __name__ == "__main__":
    run_pipeline()