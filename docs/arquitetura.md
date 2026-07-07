# Arquitetura do Sistema — Tech Challenge B

## Visão Geral

O projeto implementa um **pipeline de Machine Learning** para diagnóstico de câncer de mama com foco em segurança clínica. A arquitetura é organizada em módulos desacoplados, conectados por um pipeline central que orquestra o fluxo de dados desde a ingestão até a avaliação final.

---

## Componentes e Responsabilidades

### 1. Camada de Dados (`src/machine_learning/data/`)

| Módulo | Arquivo | Responsabilidade |
|---|---|---|
| Load | `load_data.py` | Leitura do CSV bruto a partir de `data/machine_learning/raw/` |
| Preprocess | `preprocess.py` | Limpeza, encoding, split estratificado e normalização |

**Saídas produzidas pelo Preprocess:**

```
data/machine_learning/processed/
├── X_train.csv   # features de treino (escalonadas)
├── X_val.csv     # features de validação (escalonadas)
├── X_test.csv    # features de teste (escalonadas)
├── y_train.csv   # target de treino
├── y_val.csv     # target de validação
├── y_test.csv    # target de teste
└── id_test.csv   # IDs originais dos pacientes no conjunto de teste
```

### 2. Camada de Treinamento (`src/machine_learning/training.py`)

- Recebe os splits `X_train`, `X_val`, `y_train`, `y_val` já processados
- Instancia e treina os 8 classificadores com `random_state=42` (reprodutibilidade)
- Persiste cada modelo em `models/machine_learning/<nome>.pkl` via `joblib`
- Retorna dicionário `{nome: modelo_treinado}` para a etapa de validação

### 3. Camada de Validação (`src/machine_learning/validation.py`)

- Avalia todos os modelos treinados no conjunto de **validação** (nunca no teste)
- Aplica ranking triplo: Recall → Precision → F1 da classe maligno
- Identifica e retorna `(best_name, best_model)` para a etapa de teste final

### 4. Camada de Teste Final (`src/machine_learning/test.py`)

- Executada **uma única vez**, com o melhor modelo selecionado na validação
- Produz: Accuracy, AUC-ROC, Classification Report, Confusion Matrix
- Rastreia IDs dos pacientes com falso negativo usando `id_test.csv`

### 5. Orquestrador (`src/pipeline/training_pipeline.py`)

Conecta todos os módulos em sequência sem lógica de negócio própria:

```
load_data → preprocess_data → run_training → run_validation → run_test
```

---

## Decisões Arquiteturais

### Separação em três conjuntos (64% / 16% / 20%)

A divisão em treino, validação e teste é feita em duas etapas sequenciais com `stratify=y` em ambas, garantindo que a proporção de classes (Maligno/Benigno) seja preservada nos três conjuntos.

**Motivo:** evitar que o desbalanceamento natural da base (~63% benigno, ~37% maligno) concentre casos de uma única classe em algum split, o que distorceria as métricas de avaliação.

O conjunto de **teste permanece completamente isolado** até a etapa final. Não é usado em nenhum ponto de seleção de modelo — isso garante uma avaliação honesta e imparcial da capacidade de generalização.

### StandardScaler: fit somente no treino

```python
scaler.fit_transform(X_train)   # aprende média e desvio padrão do treino
scaler.transform(X_val)         # aplica a escala do treino (sem reaprender)
scaler.transform(X_test)        # aplica a escala do treino (sem reaprender)
```

**Motivo:** aplicar `fit` em validação ou teste constituiria **data leakage** — o modelo teria acesso implícito à distribuição estatística dos dados de avaliação durante o pré-processamento. A escala aprendida no treino é a única que o modelo "conhece" e deve ser a mesma usada em produção.

### Critério primário: Recall da classe maligno

O sistema de ranking prioriza Recall sobre Accuracy ou F1 porque o custo de um **falso negativo** (classificar um tumor maligno como benigno) é clinicamente muito mais grave do que um **falso positivo** (acionar uma biópsia desnecessária). A Accuracy seria enganosa num dataset levemente desbalanceado: um modelo que classifica todos como benigno atingiria ~63% de accuracy sem detectar nenhum caso maligno.

| Critério | Ordem | Justificativa |
|---|---|---|
| Recall maligno | 1º | Minimizar falsos negativos (risco de vida) |
| Precision maligno | 2º | Desempate: reduzir biópsias desnecessárias |
| F1 maligno | 3º | Desempate final: equilíbrio geral |

### Rastreamento de falsos negativos no conjunto de teste

Os IDs originais dos pacientes são preservados **apenas para o conjunto de teste** (`id_test.csv`), e o rastreamento de falsos negativos ocorre somente na avaliação final (`test.py`).

**Motivo:** os conjuntos de treino e validação são artefatos internos do processo de aprendizado de máquina — seus erros não representam predições sobre pacientes reais, mas sim o processo de ajuste do modelo. Somente o holdout set representa predições sobre casos que o modelo nunca viu, portanto é o único conjunto para o qual identificar pacientes com diagnóstico incorreto faz sentido clinicamente.

### Persistência de modelos com joblib

Cada modelo é serializado individualmente em `.pkl` após o treinamento, em vez de manter os objetos apenas em memória.

**Motivo:** permite reutilizar modelos treinados sem re-executar o pipeline completo, possibilitar versionamento de modelos e carregar apenas o melhor modelo em produção sem manter os demais na memória.

---

## Integrações e Fluxo de Dados

```
[CSV bruto]
    │
    ▼
load_data()          → DataFrame pandas
    │
    ▼
preprocess_data()    → X_train, X_val, X_test, y_train, y_val, y_test
    │                   + salva splits em data/machine_learning/processed/
    │                   + salva id_test.csv
    ▼
run_training()       → {nome: modelo} (8 classificadores)
    │                   + salva .pkl em models/machine_learning/
    ▼
run_validation()     → (best_name, best_model)
    │                   ranking por Recall → Precision → F1
    ▼
run_test()           → {accuracy, recall_maligno, f1_maligno, falsos_negativos}
                        + rastreia IDs de falsos negativos via id_test.csv
```

---

## Estrutura de Diretórios

```
PROJETO Fase 1 - Tech challenge B/
├── data/
│   └── machine_learning/
│       ├── raw/                  # dados originais (não modificados)
│       └── processed/            # splits gerados pelo preprocess
├── docs/
│   ├── arquitetura.md            # este documento
│   └── testes.md                 # estratégia e resultados dos testes
├── models/
│   └── machine_learning/         # modelos .pkl treinados
├── notebooks/
│   └── eda.ipynb                 # análise exploratória
├── src/
│   ├── machine_learning/
│   │   ├── data/
│   │   │   ├── load_data.py
│   │   │   └── preprocess.py
│   │   ├── training.py
│   │   ├── validation.py
│   │   └── test.py
│   └── pipeline/
│       └── training_pipeline.py  # orquestrador
├── diagrama-pipeline.drawio      # diagrama de fluxo editável
└── README.md
```

---

## Tecnologias e Justificativas

| Tecnologia | Função | Justificativa |
|---|---|---|
| scikit-learn | Modelos, métricas, pré-processamento | Biblioteca padrão de ML em Python; API uniforme `fit/predict` simplifica a comparação entre algoritmos |
| pandas | Manipulação de dados tabulares | Leitura de CSV, filtro de colunas, preservação de índices para rastreamento de IDs |
| joblib | Serialização de modelos | Serialização eficiente de objetos NumPy/sklearn; recomendado pela própria scikit-learn |
| Jupyter Notebook | Análise exploratória | Visualização interativa de distribuições, correlações e padrões antes do pipeline de produção |
