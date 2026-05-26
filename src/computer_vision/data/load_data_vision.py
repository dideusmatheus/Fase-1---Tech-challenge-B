import kagglehub          # biblioteca para baixar datasets do Kaggle
import pandas as pd       # manipulação de dados tabulares
import os                 # operações de sistema de arquivos
import shutil             # cópia de arquivos entre diretórios

# Pasta local onde os CSVs do dataset serão copiados para fácil acesso
LOCAL_CSV_DIR = os.path.join("data", "computer_vision", "raw")

# ── CONSTANTES DE NOMES DE COLUNAS ────────────────────────────
# Definidas no nível do módulo para evitar repetição de strings literais

COL_IMG_PATH  = "image file path"           # coluna com caminho da mamografia completa
COL_CROP_PATH = "cropped image file path"   # coluna com caminho do recorte da região de interesse
COL_PATHOLOGY = "pathology"                 # coluna com o diagnóstico (MALIGNANT / BENIGN / ...)


def load_data_vision():
    """
    Responsabilidade: CARREGAR os metadados do dataset CBIS-DDSM.
    - Faz o download do dataset completo pelo kagglehub (usa cache se já baixou)
    - Carrega os 4 CSVs de metadados (massa e calcificação, treino e teste)
    - Combina em dois DataFrames: treino e teste
    - Retorna: (caminho_do_dataset, df_train, df_test)
    """

    # ── DOWNLOAD DO DATASET ────────────────────────────────────

    print("📥 Baixando dataset CBIS-DDSM (Breast Cancer Mammography)...")
    print("   (se já foi baixado antes, usa o cache local automaticamente)")

    # Faz o download completo do dataset para cache local no disco
    # Retorna o caminho onde os arquivos foram salvos (ex: ~/.cache/kagglehub/...)
    dataset_path = kagglehub.dataset_download(
        "awsaf49/cbis-ddsm-breast-cancer-image-dataset"  # identificador do dataset no Kaggle
    )

    # Exibe onde o dataset foi salvo localmente
    print(f"📁 Dataset disponível em: {dataset_path}")

    # ── LOCALIZA A PASTA DOS CSVs ─────────────────────────────

    # Os arquivos CSV de metadados ficam numa subpasta chamada 'csv' dentro do dataset
    csv_dir = os.path.join(dataset_path, "csv")

    # Verifica se a pasta de CSVs realmente existe antes de tentar abrir os arquivos
    if not os.path.isdir(csv_dir):
        # Se não existir, lança erro com mensagem clara para facilitar o debug
        raise FileNotFoundError(
            f"Pasta de CSVs não encontrada em: {csv_dir}\n"
            "Verifique se o dataset foi baixado corretamente."
        )

    # ── COPIA CSVs PARA PASTA LOCAL DO PROJETO ───────────────

    # Cria a pasta local caso não exista ainda
    os.makedirs(LOCAL_CSV_DIR, exist_ok=True)

    # Copia cada arquivo .csv do cache do kagglehub para data/computer_vision/raw/
    for arquivo in os.listdir(csv_dir):
        if arquivo.endswith(".csv"):
            shutil.copy2(
                os.path.join(csv_dir, arquivo),   # origem: cache do kagglehub
                os.path.join(LOCAL_CSV_DIR, arquivo)  # destino: pasta local do projeto
            )

    print(f"📄 CSVs copiados para: {LOCAL_CSV_DIR}")

    # ── CARREGA OS 4 CSVs DE METADADOS ────────────────────────

    # O dataset CBIS-DDSM tem dois tipos de anomalia: MASSA e CALCIFICAÇÃO
    # Cada tipo tem um CSV para treino e um para teste → total de 4 arquivos

    # CSV com descrições de casos de MASSA no conjunto de TREINO
    mass_train = pd.read_csv(
        os.path.join(csv_dir, "mass_case_description_train_set.csv")
    )

    # CSV com descrições de casos de MASSA no conjunto de TESTE
    mass_test = pd.read_csv(
        os.path.join(csv_dir, "mass_case_description_test_set.csv")
    )

    # CSV com descrições de casos de CALCIFICAÇÃO no conjunto de TREINO
    calc_train = pd.read_csv(
        os.path.join(csv_dir, "calc_case_description_train_set.csv")
    )

    # CSV com descrições de casos de CALCIFICAÇÃO no conjunto de TESTE
    calc_test = pd.read_csv(
        os.path.join(csv_dir, "calc_case_description_test_set.csv")
    )

    # ── SELECIONA APENAS AS COLUNAS NECESSÁRIAS ───────────────

    # Lista com as colunas de interesse para a CNN
    cols_desejadas = [COL_IMG_PATH, COL_CROP_PATH, COL_PATHOLOGY]

    def selecionar_colunas(df, colunas):
        # Filtra apenas as colunas que existem no DataFrame (proteção contra versões diferentes do CSV)
        colunas_disponiveis = [c for c in colunas if c in df.columns]
        # Retorna uma cópia com apenas essas colunas para não alterar o DataFrame original
        return df[colunas_disponiveis].copy()

    # Aplica a seleção segura nos 4 DataFrames
    mass_train = selecionar_colunas(mass_train, cols_desejadas)
    mass_test  = selecionar_colunas(mass_test,  cols_desejadas)
    calc_train = selecionar_colunas(calc_train, cols_desejadas)
    calc_test  = selecionar_colunas(calc_test,  cols_desejadas)

    # ── COMBINA MASSA + CALCIFICAÇÃO ──────────────────────────

    # Une os casos de massa e calcificação no conjunto de TREINO
    # ignore_index=True: reinicia o índice de 0 no DataFrame resultante
    df_train = pd.concat([mass_train, calc_train], ignore_index=True)

    # Une os casos de massa e calcificação no conjunto de TESTE
    df_test = pd.concat([mass_test, calc_test], ignore_index=True)

    # ── LIMPEZA BÁSICA ────────────────────────────────────────

    # Remove linhas onde o diagnóstico está ausente (não há como treinar sem rótulo)
    df_train = df_train.dropna(subset=[COL_PATHOLOGY])
    df_test  = df_test.dropna(subset=[COL_PATHOLOGY])

    # Remove linhas onde o caminho da imagem está ausente (não há como carregar sem caminho)
    df_train = df_train.dropna(subset=[COL_IMG_PATH])
    df_test  = df_test.dropna(subset=[COL_IMG_PATH])

    # Remove duplicatas baseando-se no caminho da imagem
    # O mesmo exame pode aparecer em múltiplas linhas (ex: diferentes anotadores)
    df_train = df_train.drop_duplicates(subset=[COL_IMG_PATH])
    df_test  = df_test.drop_duplicates(subset=[COL_IMG_PATH])

    # Reinicia os índices após todas as remoções para índice contínuo de 0 a N
    df_train = df_train.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    # ── RESUMO ────────────────────────────────────────────────

    print("\n✅ Metadados carregados com sucesso:")
    print(f"   Treino : {len(df_train)} casos")
    print(f"   Teste  : {len(df_test)} casos")

    # Exibe a distribuição das classes no treino para verificar desbalanceamento
    print("\n   Distribuição de classes (treino):")
    for classe, count in df_train[COL_PATHOLOGY].value_counts().items():
        print(f"   → {classe}: {count} casos")

    # Retorna:
    # dataset_path → necessário para construir os caminhos das imagens
    # df_train     → metadados do conjunto de treino
    # df_test      → metadados do conjunto de teste
    return dataset_path, df_train, df_test


if __name__ == "__main__":
    # Execução direta: baixa e carrega os dados para verificação rápida
    dataset_path, df_train, df_test = load_data_vision()

    # Exibe as primeiras linhas do treino para inspeção visual
    print("\n📋 Primeiras linhas do conjunto de treino:")
    print(df_train.head())

    # Exibe as primeiras linhas do teste para inspeção visual
    print("\n📋 Primeiras linhas do conjunto de teste:")
    print(df_test.head())

    # Exibe as colunas disponíveis para confirmar a estrutura
    print(f"\n📐 Colunas disponíveis: {df_train.columns.tolist()}")
