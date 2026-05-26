import os                                   # operações de sistema de arquivos
import numpy as np                          # operações numéricas
import tensorflow as tf                     # framework de deep learning
from sklearn.model_selection import train_test_split  # divisão treino/validação

# ── CONSTANTES DE CONFIGURAÇÃO ────────────────────────────────
# Centralizadas aqui para facilitar ajuste sem precisar editar múltiplos arquivos

IMG_SIZE     = (224, 224)  # tamanho padrão do MobileNetV2: 224x224 pixels
IMG_CHANNELS = 3           # número de canais: 3 = RGB (mamografias serão convertidas para RGB)
BATCH_SIZE   = 32          # número de imagens processadas por vez na GPU/CPU
VAL_SIZE     = 0.2         # 20% dos dados de treino reservados para validação


def construir_caminho_imagem(dataset_path, caminho_csv):
    """
    Converte o caminho relativo do CSV para o caminho completo no disco.

    O CSV contém caminhos no formato DICOM, ex:
    'Mass-Training_P_00001_LEFT_CC/{UID_serie}/{UID_instancia}/000000.dcm'

    No dataset JPEG do Kaggle (awsaf49), a estrutura real é:
    '{dataset_path}/jpeg/{UID_instancia}/{arquivo}.jpg'

    - O UID_instancia é o penúltimo segmento do caminho CSV (partes[-2])
    - O nome do arquivo JPEG não é '000000.jpg'; pode ser '1-211.jpg' etc.
    - Cada pasta de instância contém exatamente 1 arquivo .jpg
    """

    # Normaliza separadores de pasta (Windows usa \\, Unix usa /)
    caminho_normalizado = str(caminho_csv).strip().replace("\\", "/")

    # Divide o caminho em segmentos separados por '/'
    partes = caminho_normalizado.split("/")

    # O penúltimo segmento é o UID_instancia, que nomeia a pasta no dataset JPEG
    uid_instancia = partes[-2]

    # Monta o caminho da pasta que contém a imagem JPEG desta instância
    pasta_imagem = os.path.join(dataset_path, "jpeg", uid_instancia)

    # Se a pasta não existe, retorna o caminho da pasta (falhará no os.path.exists)
    if not os.path.isdir(pasta_imagem):
        return pasta_imagem

    # Lista os arquivos .jpg dentro da pasta (normalmente há apenas 1)
    arquivos_jpg = [f for f in os.listdir(pasta_imagem) if f.lower().endswith(".jpg")]

    # Se não há nenhum .jpg, retorna a pasta (falhará no os.path.exists)
    if not arquivos_jpg:
        return pasta_imagem

    # Retorna o caminho completo do primeiro (e geralmente único) .jpg encontrado
    return os.path.join(pasta_imagem, arquivos_jpg[0])


def codificar_labels(df):
    """
    Converte a coluna 'pathology' de texto para rótulo binário.

    MALIGNANT                 → 1  (caso positivo: câncer detectado)
    BENIGN                    → 0  (caso negativo: sem câncer)
    BENIGN_WITHOUT_CALLBACK   → 0  (benigno confirmado, sem necessidade de retorno)
    """

    # Cria uma cópia para não modificar o DataFrame original passado como argumento
    df = df.copy()

    # Aplica o mapeamento: MALIGNANT vira 1, qualquer outro valor vira 0
    # strip() remove espaços acidentais no início/fim da string
    df["label"] = df["pathology"].str.strip().apply(
        lambda x: 1 if x == "MALIGNANT" else 0
    )

    return df


def adicionar_caminhos_completos(dataset_path, df):
    """
    Adiciona a coluna 'full_path' ao DataFrame com o caminho absoluto de cada imagem.
    Remove linhas cujo arquivo de imagem não existe no disco.

    Usa 'cropped image file path' se disponível (recorte da lesão, mais informativo),
    senão usa 'image file path' (mamografia completa).
    """

    # Decide qual coluna de caminho usar: prefere o recorte (crop) da lesão
    if "cropped image file path" in df.columns:
        col_path = "cropped image file path"  # recorte foca na região da anomalia
    else:
        col_path = "image file path"          # mamografia completa como fallback

    # Constrói o caminho completo para cada linha do DataFrame
    df = df.copy()                            # cópia para não alterar o original
    df["full_path"] = df[col_path].apply(
        lambda p: construir_caminho_imagem(dataset_path, p)  # aplica a conversão linha a linha
    )

    # Conta quantas imagens existem antes da filtragem
    total_antes = len(df)

    # Remove linhas cujo arquivo de imagem não existe no disco
    # Imagem inexistente causaria erro durante o treinamento
    df = df[df["full_path"].apply(os.path.exists)].reset_index(drop=True)

    # Conta quantas imagens foram removidas por não existirem
    removidas = total_antes - len(df)
    if removidas > 0:
        # Avisa o usuário sobre imagens não encontradas (pode indicar problema no dataset)
        print(f"⚠️  {removidas} imagens não encontradas no disco e removidas.")

    return df


def carregar_e_preprocessar_imagem(caminho, label):
    """
    Função aplicada a cada item do tf.data.Dataset:
    - Lê o arquivo JPEG do disco
    - Decodifica para tensor de imagem
    - Redimensiona para IMG_SIZE (224x224)
    - Normaliza valores de pixel para [0.0, 1.0]
    Retorna o par (imagem_processada, rótulo) para o modelo.
    """

    # Lê o arquivo de imagem em formato de bytes brutos
    bytes_imagem = tf.io.read_file(caminho)

    # Decodifica os bytes como imagem JPEG com o número de canais definido (3 = RGB)
    # Mamografias são originalmente em escala de cinza mas MobileNetV2 espera 3 canais
    imagem = tf.image.decode_jpeg(bytes_imagem, channels=IMG_CHANNELS)

    # Redimensiona a imagem para o tamanho esperado pelo modelo (224x224 pixels)
    # Método padrão: interpolação bilinear (equilibra qualidade e velocidade)
    imagem = tf.image.resize(imagem, IMG_SIZE)

    # Converte os valores de pixel de inteiros [0, 255] para float [0.0, 1.0]
    # Normalização facilita o aprendizado: gradientes ficam em escala uniforme
    imagem = tf.cast(imagem, tf.float32) / 255.0

    # Retorna o par (imagem, rótulo) que o Keras espera durante o treinamento
    return imagem, label


def aumentar_imagem(imagem, label):
    """
    Aplica transformações aleatórias nas imagens de treino (Data Augmentation).
    Objetivo: artificialmente aumentar a diversidade dos dados de treino,
    tornando o modelo mais robusto e reduzindo overfitting.

    Apenas usado no conjunto de TREINO, nunca na validação ou teste.
    """

    # Espelhamento horizontal aleatório (50% de chance)
    # Simula mamografias do lado esquerdo e direito indistintamente
    imagem = tf.image.random_flip_left_right(imagem)

    # Espelhamento vertical aleatório (50% de chance)
    # Aumenta variedade de orientações sem distorcer a anatomia
    imagem = tf.image.random_flip_up_down(imagem)

    # Variação aleatória de brilho (±10%)
    # Simula diferenças na exposição e equipamento radiológico
    imagem = tf.image.random_brightness(imagem, max_delta=0.1)

    # Variação aleatória de contraste (fator entre 0.9 e 1.1)
    # Simula diferenças na densidade do tecido mamário
    imagem = tf.image.random_contrast(imagem, lower=0.9, upper=1.1)

    # Garante que após as transformações os valores continuem em [0.0, 1.0]
    # Evita valores negativos ou acima de 1 que quebrariam o treinamento
    imagem = tf.clip_by_value(imagem, 0.0, 1.0)

    return imagem, label


def criar_dataset(caminhos, labels, aumentar=False, embaralhar=False):
    """
    Cria um tf.data.Dataset otimizado para treino ou avaliação.

    caminhos   → lista de caminhos completos das imagens
    labels     → lista de rótulos binários correspondentes (0 ou 1)
    aumentar   → True somente no treino (aplica data augmentation)
    embaralhar → True somente no treino (evita viés de ordem)
    """

    # Converte as listas Python para tensores TensorFlow
    tensor_caminhos = tf.constant(caminhos, dtype=tf.string)   # strings dos caminhos
    tensor_labels   = tf.constant(labels,   dtype=tf.int32)    # inteiros 0 ou 1

    # Cria o dataset a partir dos pares (caminho, label)
    dataset = tf.data.Dataset.from_tensor_slices((tensor_caminhos, tensor_labels))

    # Embaralha os dados antes do treinamento para evitar padrões de ordem
    # buffer_size=len(caminhos) garante embaralhamento completo
    if embaralhar:
        dataset = dataset.shuffle(buffer_size=len(caminhos), seed=42)

    # Aplica o carregamento e pré-processamento de cada imagem
    # AUTOTUNE: TensorFlow decide automaticamente o nível de paralelismo ideal
    dataset = dataset.map(carregar_e_preprocessar_imagem, num_parallel_calls=tf.data.AUTOTUNE)

    # Aplica data augmentation somente se solicitado (apenas no treino)
    if aumentar:
        dataset = dataset.map(aumentar_imagem, num_parallel_calls=tf.data.AUTOTUNE)

    # Agrupa as imagens em batches do tamanho definido em BATCH_SIZE
    dataset = dataset.batch(BATCH_SIZE)

    # Pré-carrega o próximo batch enquanto o modelo processa o atual
    # Evita que GPU/CPU fique ociosa esperando dados do disco
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    return dataset


def calcular_pesos_classes(labels_treino):
    """
    Calcula pesos inversamente proporcionais à frequência de cada classe.

    No CBIS-DDSM, casos benignos são mais frequentes que malignos.
    Sem correção, o modelo aprenderia a sempre prever 'benigno' para
    ter alta accuracy, mas com muitos falsos negativos (canceres perdidos).

    Fórmula: peso_classe = total / (n_classes * frequencia_classe)
    """

    # Conta quantas amostras de cada classe existem no treino
    n_maligno = int(np.sum(labels_treino))          # rótulo 1 = maligno
    n_benigno = len(labels_treino) - n_maligno      # rótulo 0 = benigno
    n_total   = len(labels_treino)                  # total de amostras

    # Calcula o peso de cada classe: quanto mais rara, maior o peso
    peso_benigno  = n_total / (2.0 * n_benigno)    # peso para classe 0
    peso_maligno  = n_total / (2.0 * n_maligno)    # peso para classe 1

    # Retorna dicionário no formato esperado pelo Keras: {classe: peso}
    return {0: peso_benigno, 1: peso_maligno}


def preprocess_data_vision(dataset_path, df_train, df_test):
    """
    Pipeline completo de pré-processamento para visão computacional.

    Etapas:
    1. Codifica rótulos (texto → 0/1)
    2. Constrói caminhos completos e valida existência dos arquivos
    3. Divide dados de treino em treino + validação (80/20)
    4. Calcula pesos de classe para balancear o treinamento
    5. Cria tf.data.Dataset para cada split

    Retorna: (train_ds, val_ds, test_ds, pesos_classes)
    """

    # ── CODIFICA RÓTULOS ──────────────────────────────────────

    # Converte 'pathology' de texto para label binário (0 ou 1)
    df_train = codificar_labels(df_train)
    df_test  = codificar_labels(df_test)

    # ── CONSTRÓI CAMINHOS COMPLETOS ───────────────────────────

    print("🔍 Localizando imagens no disco...")

    # Adiciona coluna 'full_path' com caminho absoluto e remove imagens inexistentes
    df_train = adicionar_caminhos_completos(dataset_path, df_train)
    df_test  = adicionar_caminhos_completos(dataset_path, df_test)

    # ── DIVIDE TREINO EM TREINO + VALIDAÇÃO ───────────────────

    # Separa os caminhos e labels do treino em listas Python
    caminhos_treino = df_train["full_path"].tolist()   # lista de caminhos
    labels_treino   = df_train["label"].tolist()       # lista de 0s e 1s

    # Divide em 80% treino e 20% validação
    # stratify garante que a proporção maligno/benigno seja igual nos dois splits
    caminhos_tr, caminhos_val, labels_tr, labels_val = train_test_split(
        caminhos_treino,       # caminhos das imagens de treino
        labels_treino,         # rótulos correspondentes
        test_size=VAL_SIZE,    # 20% para validação
        random_state=42,       # semente fixa para reprodutibilidade
        stratify=labels_treino # mantém proporção das classes
    )

    # Extrai caminhos e labels do conjunto de TESTE
    caminhos_test = df_test["full_path"].tolist()
    labels_test   = df_test["label"].tolist()

    # ── CALCULA PESOS DE CLASSE ───────────────────────────────

    # Pesos inversamente proporcionais à frequência para lidar com desbalanceamento
    pesos_classes = calcular_pesos_classes(labels_tr)

    print("\n⚖️  Pesos das classes (balanceamento do treino):")
    print(f"   Benigno  (0): {pesos_classes[0]:.4f}")
    print(f"   Maligno  (1): {pesos_classes[1]:.4f}")

    # ── CRIA tf.data.Dataset PARA CADA SPLIT ─────────────────

    print("\n📦 Criando tf.data.Datasets...")

    # Dataset de TREINO: com embaralhamento e data augmentation
    train_ds = criar_dataset(
        caminhos_tr, labels_tr,
        aumentar=True,     # aplica flip, brilho, contraste aleatórios
        embaralhar=True    # embaralha para evitar viés de ordem nos batches
    )

    # Dataset de VALIDAÇÃO: sem embaralhamento, sem augmentation
    val_ds = criar_dataset(
        caminhos_val, labels_val,
        aumentar=False,    # avaliação deve ser determinística
        embaralhar=False   # ordem fixa para comparação consistente entre épocas
    )

    # Dataset de TESTE: sem embaralhamento, sem augmentation
    test_ds = criar_dataset(
        caminhos_test, labels_test,
        aumentar=False,    # avaliação final deve ser determinística
        embaralhar=False   # ordem fixa para rastrear predições individuais
    )

    # ── RESUMO ────────────────────────────────────────────────

    print("\n✅ Pré-processamento concluído:")
    print(f"   Treino    : {len(caminhos_tr):>5} imagens  ({IMG_SIZE[0]}x{IMG_SIZE[1]} px, RGB)")
    print(f"   Validação : {len(caminhos_val):>5} imagens")
    print(f"   Teste     : {len(caminhos_test):>5} imagens")
    print(f"   Batch size: {BATCH_SIZE}")

    # Retorna os três datasets e os pesos de classe para uso no treinamento
    return train_ds, val_ds, test_ds, pesos_classes
