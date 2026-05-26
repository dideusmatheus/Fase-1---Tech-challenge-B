import sys   # manipulação do caminho de busca de módulos Python
import os    # operações de sistema de arquivos

# Adiciona a raiz do projeto ao Python path para que os imports funcionem
# independente de onde o script é chamado
# __file__ = caminho deste arquivo (src/pipeline/training_pipeline_vision.py)
# os.path.dirname(__file__) = pasta src/pipeline/
# os.path.join(..., "../..") = sobe dois níveis = raiz do projeto
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

# Importa as funções de cada etapa do pipeline de visão computacional
from src.computer_vision.data.load_data_vision  import load_data_vision      # etapa 1: carga
from src.computer_vision.data.preprocess_vision import preprocess_data_vision # etapa 2: pré-processamento
from src.computer_vision.training               import run_training_vision    # etapa 3: treinamento
from src.computer_vision.validation             import run_validation_vision  # etapa 4: validação
from src.computer_vision.test                   import run_test_vision        # etapa 5: teste final


def run_pipeline_vision():
    """
    Pipeline completo de Visão Computacional — CNN para diagnóstico de mamografia.

    Fluxo:
      1. Carrega os metadados do dataset CBIS-DDSM (download via kagglehub)
      2. Pré-processa: constrói caminhos das imagens, codifica labels,
         cria tf.data.Dataset para treino/validação/teste
      3. Treina a CNN com Transfer Learning (MobileNetV2) em duas fases:
         Fase 1 — base congelada (treina apenas a cabeça)
         Fase 2 — fine-tuning das últimas camadas da base
      4. Valida a CNN no conjunto de validação para monitorar a qualidade
      5. Avalia o modelo no conjunto de teste (executa UMA única vez)

    Retorna: dicionário com as métricas finais do teste
    """

    # ── ETAPA 1: CARREGAMENTO DOS DADOS ───────────────────────

    print("\n" + "="*55)
    print("📂 ETAPA 1 — CARREGAMENTO DOS DADOS")
    print("="*55)

    # Faz o download do dataset CBIS-DDSM e carrega os CSVs de metadados
    # Retorna: caminho local do dataset, DataFrame de treino, DataFrame de teste
    dataset_path, df_train, df_test = load_data_vision()

    # ── ETAPA 2: PRÉ-PROCESSAMENTO ────────────────────────────

    print("\n" + "="*55)
    print("🔧 ETAPA 2 — PRÉ-PROCESSAMENTO DAS IMAGENS")
    print("="*55)

    # Constrói os caminhos completos das imagens, codifica labels binários,
    # divide treino em treino/validação e cria os tf.data.Datasets
    # Retorna: train_ds, val_ds, test_ds (tf.data.Dataset), pesos_classes (dict)
    train_ds, val_ds, test_ds, pesos_classes = preprocess_data_vision(
        dataset_path,  # caminho raiz do dataset baixado pelo kagglehub
        df_train,      # DataFrame com metadados do treino
        df_test        # DataFrame com metadados do teste
    )

    # ── ETAPA 3: TREINAMENTO DA CNN ───────────────────────────

    print("\n" + "="*55)
    print("🏋️  ETAPA 3 — TREINAMENTO DA CNN")
    print("="*55)

    # Treina a CNN com Transfer Learning em duas fases
    # Os históricos são ignorados aqui (use o notebook eda.ipynb para plotar as curvas)
    modelo, _, _ = run_training_vision(
        train_ds,       # dataset de treino com augmentation
        val_ds,         # dataset de validação para monitorar o treino
        pesos_classes   # pesos para compensar desbalanceamento de classes
    )

    # ── ETAPA 4: VALIDAÇÃO ────────────────────────────────────

    print("\n" + "="*55)
    print("🔍 ETAPA 4 — VALIDAÇÃO DA CNN")
    print("="*55)

    # Avalia o modelo no conjunto de validação e exibe métricas intermediárias
    # Retorna: dicionário com accuracy, auc, recall, precision, f1, confusion_matrix
    metricas_val = run_validation_vision(modelo, val_ds)

    # Exibe o recall de maligno na validação como resumo da etapa
    print(f"\n   → Recall Maligno (validação): {metricas_val['recall_maligno']:.4f}")

    # ── ETAPA 5: AVALIAÇÃO FINAL NO TESTE ────────────────────

    print("\n" + "="*55)
    print("🏁 ETAPA 5 — AVALIAÇÃO FINAL NO TESTE")
    print("="*55)

    # Avalia o modelo no conjunto de TESTE (executado uma única vez)
    # Este é o resultado honesto e definitivo do modelo
    metricas_teste = run_test_vision(modelo, test_ds)

    # Retorna as métricas finais para uso externo (ex: notebook de análise)
    return metricas_teste


if __name__ == "__main__":
    # Ponto de entrada principal: executa o pipeline completo de visão computacional
    # Para rodar: python src/pipeline/training_pipeline_vision.py
    metricas = run_pipeline_vision()

    # Exibe as métricas finais em formato resumido
    print("\n📊 Métricas finais retornadas pelo pipeline:")
    for chave, valor in metricas.items():
        # Pula a matriz de confusão (exibida separadamente pelo test.py)
        if chave != "confusion_matrix":
            print(f"   {chave}: {valor}")
