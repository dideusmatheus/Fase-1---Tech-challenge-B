import os                    # operações de sistema de arquivos
import tensorflow as tf      # framework de deep learning

# ── CONSTANTES DE CONFIGURAÇÃO ────────────────────────────────

IMG_SIZE     = (224, 224)  # resolução esperada pelo MobileNetV2
IMG_CHANNELS = 3           # canais RGB (3 = colorido)
EPOCHS       = 20          # máximo de épocas na fase de transfer learning
FINE_TUNE_EPOCHS = 10      # máximo de épocas na fase de fine-tuning


def construir_modelo():
    """
    Constrói a CNN usando Transfer Learning com MobileNetV2.

    Arquitetura em duas partes:
    1. BASE: MobileNetV2 pré-treinada no ImageNet (congelada inicialmente)
       - Já aprendeu a reconhecer bordas, texturas e formas com 1.2M imagens
       - Reutilizamos esse conhecimento para o domínio médico (mamografias)

    2. CABEÇA: camadas densas customizadas para classificação binária
       - GlobalAveragePooling2D → compacta features espaciais
       - Dense(128, relu)       → aprende combinações das features
       - Dropout(0.3)           → regularização para evitar overfitting
       - Dense(1, sigmoid)      → probabilidade de ser MALIGNO (0.0 a 1.0)

    Por que MobileNetV2?
    - Leve: treinável em CPU em tempo razoável
    - Preciso: bom trade-off entre tamanho e acurácia
    - Ideal para projetos educacionais e datasets médicos pequenos
    """

    # ── BASE PRÉ-TREINADA ─────────────────────────────────────

    # Carrega o MobileNetV2 com pesos do ImageNet (treinado em 1.2M imagens)
    # include_top=False: remove as camadas de classificação originais (1000 classes)
    # Manteremos apenas o extrator de features (convolucional)
    base = tf.keras.applications.MobileNetV2(
        input_shape=(*IMG_SIZE, IMG_CHANNELS),  # (224, 224, 3) — formato de entrada
        include_top=False,                       # sem a cabeça original do ImageNet
        weights="imagenet"                       # carrega pesos pré-treinados
    )

    # Congela todos os pesos da base durante a Fase 1
    # Objetivo: treinar apenas a cabeça nova sem destruir o que a base já aprendeu
    base.trainable = False

    # ── ENTRADA ───────────────────────────────────────────────

    # Define o formato de entrada do modelo (224x224 pixels, 3 canais RGB)
    entradas = tf.keras.Input(shape=(*IMG_SIZE, IMG_CHANNELS))

    # ── EXTRAÇÃO DE FEATURES ──────────────────────────────────

    # Passa as imagens pela base MobileNetV2 para extrair features
    # training=False: mantém BatchNorm em modo inferência mesmo durante o treino
    # Isso evita que a normalização seja afetada pelos dados médicos durante a Fase 1
    x = base(entradas, training=False)

    # ── CABEÇA DE CLASSIFICAÇÃO ───────────────────────────────

    # GlobalAveragePooling2D: transforma o mapa de features (7x7xC) em vetor (C,)
    # Equivale a tirar a média espacial de cada filtro → compacta sem perder informação
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    # Dropout 30%: durante o treino, desativa aleatoriamente 30% dos neurônios
    # Força o modelo a não depender de neurônios específicos → reduz overfitting
    x = tf.keras.layers.Dropout(rate=0.3)(x)

    # Camada densa com 128 neurônios e ativação ReLU
    # Aprende combinações não-lineares das features extraídas pela base
    x = tf.keras.layers.Dense(128, activation="relu")(x)

    # Segundo Dropout antes da saída final (regularização adicional)
    x = tf.keras.layers.Dropout(rate=0.2)(x)

    # Camada de saída: 1 neurônio com sigmoid
    # sigmoid: mapeia qualquer valor real para [0, 1] = probabilidade de ser MALIGNO
    # Threshold padrão: >= 0.5 → MALIGNO (1), < 0.5 → BENIGNO (0)
    saidas = tf.keras.layers.Dense(1, activation="sigmoid")(x)

    # Monta o modelo completo conectando entradas e saídas
    modelo = tf.keras.Model(entradas, saidas, name="cnn_mamografia_mobilenetv2")

    # Retorna o modelo E a referência à base (necessária para o fine-tuning)
    return modelo, base


def compilar_modelo(modelo, learning_rate=1e-3):
    """
    Configura o processo de treinamento do modelo.

    learning_rate: quanto os pesos são ajustados a cada batch
    - Fase 1 (base congelada): 1e-3 (maior, pois só a cabeça é treinada)
    - Fase 2 (fine-tuning): 1e-5 (menor, para não destruir pesos pré-treinados)
    """

    # Compila o modelo com otimizador, função de perda e métricas
    modelo.compile(
        # Adam: otimizador adaptativo, padrão para redes neurais profundas
        # Ajusta o learning rate individualmente para cada parâmetro
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),

        # Binary crossentropy: loss padrão para classificação binária
        # Penaliza mais quando a previsão está errada e com alta confiança
        loss="binary_crossentropy",

        # Métricas monitoradas durante treino e validação
        metrics=[
            "accuracy",                                  # % geral de acertos
            tf.keras.metrics.AUC(name="auc"),            # área sob a curva ROC
            tf.keras.metrics.Recall(name="recall"),      # sensibilidade (MÉTRICA PRINCIPAL)
            tf.keras.metrics.Precision(name="precision") # precisão
        ]
    )

    return modelo


def run_training_vision(train_ds, val_ds, pesos_classes):
    """
    Responsabilidade: TREINAR a CNN em duas fases.

    Fase 1 — Transfer Learning (base congelada):
    - Treina apenas a cabeça de classificação customizada
    - Learning rate maior (1e-3)
    - Rápida convergência

    Fase 2 — Fine-tuning (últimas camadas da base descongeladas):
    - Refina as camadas mais especializadas da base MobileNetV2
    - Learning rate muito menor (1e-5) para não destruir pesos pré-treinados
    - Melhora a adaptação ao domínio médico (mamografias)

    pesos_classes: dict {0: peso_benigno, 1: peso_maligno}
    Corrige o desbalanceamento entre classes durante o cálculo da loss
    """

    # Cria a pasta de destino para salvar os modelos treinados
    os.makedirs("models/computer_vision", exist_ok=True)

    # ── FASE 1: TRANSFER LEARNING ─────────────────────────────

    print("\n" + "="*55)
    print("🧠 CNN — FASE 1: TRANSFER LEARNING")
    print("   Base MobileNetV2 congelada | treina só a cabeça")
    print("="*55)

    # Constrói o modelo com a base MobileNetV2 congelada
    modelo, base = construir_modelo()

    # Configura o treinamento com learning rate para a Fase 1
    modelo = compilar_modelo(modelo, learning_rate=1e-3)

    # Exibe o resumo da arquitetura (número de parâmetros treináveis vs. congelados)
    modelo.summary()

    # ── CALLBACKS ─────────────────────────────────────────────
    # Callbacks são ações automáticas executadas pelo Keras durante o treino

    # Salva automaticamente o modelo com a menor val_loss encontrada até o momento
    cb_checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/computer_vision/cnn_melhor.keras",  # caminho do arquivo
        monitor="val_loss",       # monitora a perda na validação
        save_best_only=True,      # salva apenas se melhorou
        verbose=1                 # exibe mensagem quando salva
    )

    # Interrompe o treino se val_loss não melhorar por 5 épocas consecutivas
    # restore_best_weights=True: ao parar, restaura os melhores pesos encontrados
    cb_early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",         # monitora a perda na validação
        patience=5,                 # aguarda até 5 épocas sem melhora
        restore_best_weights=True,  # volta para o estado de menor val_loss
        verbose=1
    )

    # Reduz o learning rate pela metade se val_loss não melhorar por 3 épocas
    # Evita que o otimizador "pule" o mínimo da função de perda
    cb_reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",   # monitora a perda na validação
        factor=0.5,           # multiplica o LR atual por 0.5 quando aciona
        patience=3,           # aguarda 3 épocas antes de reduzir
        min_lr=1e-7,          # não reduz abaixo deste valor mínimo
        verbose=1
    )

    # ── TREINO FASE 1 ─────────────────────────────────────────

    print(f"\n⏳ Iniciando Fase 1 — até {EPOCHS} épocas...")

    # Executa o treinamento com os datasets e callbacks configurados
    historico_1 = modelo.fit(
        train_ds,                                          # dataset de treino
        validation_data=val_ds,                            # dataset de validação
        epochs=EPOCHS,                                     # máximo de épocas
        class_weight=pesos_classes,                        # corrige desbalanceamento
        callbacks=[cb_checkpoint, cb_early_stop, cb_reduce_lr],  # callbacks ativos
        verbose=1                                          # exibe progresso por época
    )

    # ── FASE 2: FINE-TUNING ───────────────────────────────────

    print("\n" + "="*55)
    print("🔧 CNN — FASE 2: FINE-TUNING")
    print("   Descongela últimas 30% da base | LR reduzido")
    print("="*55)

    # Desbloqueia os pesos da base para que possam ser ajustados
    base.trainable = True

    # Total de camadas na base MobileNetV2
    total_camadas = len(base.layers)

    # Mantém as primeiras 70% das camadas congeladas
    # Essas camadas aprenderam features genéricas (bordas, curvas) que são reutilizáveis
    # Apenas as últimas 30% (features de alto nível) serão refinadas
    inicio_finetune = int(total_camadas * 0.7)  # índice onde começa o descongelamento

    for camada in base.layers[:inicio_finetune]:
        camada.trainable = False   # mantém congelada: feature genérica

    for camada in base.layers[inicio_finetune:]:
        camada.trainable = True    # descongela para fine-tuning: feature especializada

    print(f"   Camadas totais na base   : {total_camadas}")
    print(f"   Camadas descongeladas    : {total_camadas - inicio_finetune} (últimas 30%)")

    # Recompila com learning rate 100x menor que a Fase 1
    # LR menor é essencial para não destruir os pesos pré-treinados que já funcionam
    modelo = compilar_modelo(modelo, learning_rate=1e-5)

    # Checkpoint para o fine-tuning (sobrescreve se o novo modelo for melhor)
    cb_checkpoint_ft = tf.keras.callbacks.ModelCheckpoint(
        filepath="models/computer_vision/cnn_melhor.keras",  # mesmo arquivo
        monitor="val_loss",
        save_best_only=True,
        verbose=1
    )

    # Early stopping também para o fine-tuning
    cb_early_stop_ft = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    print(f"\n⏳ Iniciando Fase 2 — até {FINE_TUNE_EPOCHS} épocas...")

    # Executa o fine-tuning com learning rate reduzido
    historico_2 = modelo.fit(
        train_ds,
        validation_data=val_ds,
        epochs=FINE_TUNE_EPOCHS,
        class_weight=pesos_classes,
        callbacks=[cb_checkpoint_ft, cb_early_stop_ft],
        verbose=1
    )

    # ── SALVA O MODELO FINAL ──────────────────────────────────

    # Salva o modelo completo (arquitetura + pesos) no formato Keras nativo
    modelo.save("models/computer_vision/cnn_final.keras")
    print("\n💾 Modelo final salvo em models/computer_vision/cnn_final.keras")

    print("\n" + "="*55)
    print("✅ Treinamento concluído — ambas as fases!")
    print("="*55)

    # Retorna o modelo treinado e os históricos de ambas as fases para análise
    return modelo, historico_1, historico_2
