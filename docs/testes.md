# Estratégia de Testes — Tech Challenge B

## Abordagem Geral

O projeto adota uma **estratégia de validação por holdout**, com três conjuntos separados e responsabilidades bem definidas:

| Conjunto | Tamanho | Papel |
|---|---|---|
| Treino (X_train) | 64% | Ajuste dos parâmetros dos modelos (`fit`) |
| Validação (X_val) | 16% | Seleção do melhor modelo (nunca influencia o treino) |
| Teste (X_test) | 20% | Avaliação final do desempenho real — executada **uma única vez** |

A separação estratificada (`stratify=y`) garante que a proporção de classes (maligno/benigno) seja preservada nos três conjuntos, evitando viés na avaliação causado pelo leve desbalanceamento da base (~63% benigno, ~37% maligno).

---

## Etapa 1 — Validação de Pré-processamento

**Arquivo:** `src/machine_learning/data/preprocess.py`

Verificações aplicadas:

- Remoção da coluna `id` (sem valor preditivo)
- Remoção de colunas com taxa de valores ausentes > 40%
- Remoção de linhas duplicadas
- Label Encoding: `M → 1`, `B → 0`; linhas com target fora desse mapeamento são descartadas

**Anti-data-leakage:**

O `StandardScaler` executa `fit_transform` exclusivamente em `X_train`. Em `X_val` e `X_test` aplica-se apenas `transform`, usando os parâmetros (média e desvio padrão) aprendidos no treino. Isso garante que nenhuma informação dos conjuntos de avaliação contamina o processo de aprendizado.

---

## Etapa 2 — Avaliação dos Modelos no Conjunto de Validação

**Arquivo:** `src/machine_learning/validation.py`

Os 8 modelos são avaliados com as seguintes métricas, calculadas no conjunto de validação:

| Métrica | Descrição | Prioridade |
|---|---|---|
| **Recall (Maligno)** | % dos casos malignos corretamente detectados | 1ª (critério principal) |
| **Precision (Maligno)** | % das predições "maligno" que estavam corretas | 2ª (desempate) |
| **F1-score (Maligno)** | Média harmônica entre Recall e Precision | 3ª (desempate final) |
| AUC-ROC | Área sob a curva ROC — independente do threshold | Apoio |
| Accuracy | % geral de acertos | Informativo |

O Recall é o critério primário porque um **falso negativo** (tumor maligno classificado como benigno) representa risco de vida para o paciente. Um modelo com 100% de Recall mas Precision baixa é preferível a um com alta Accuracy e Recall inferior.

**Ranking de seleção:**

```python
sorted(recall_scores.items(),
       key=lambda x: (recall[x], precision[x], f1[x]),
       reverse=True)
```

---

## Etapa 3 — Avaliação Final no Holdout Set

**Arquivo:** `src/machine_learning/test.py`

Executada **uma única vez** com o melhor modelo identificado na validação. Qualquer re-execução após visualizar os resultados invalidaria a avaliação (overfitting manual ao conjunto de teste).

**Métricas reportadas:**

```
Accuracy        : 0.9737
AUC-ROC         : (calculado se modelo suporta predict_proba)
Recall Maligno  : 0.9524  ← métrica principal
Precision Malig.: (reportado no classification report)
F1 Maligno      : 0.9639
Falsos Negativos: 2
```

**Matriz de confusão:**

```
                   Previsto Benigno   Previsto Maligno
Real Benigno       TN (correto)       FP (alarme falso)
Real Maligno       FN ⚠️ (perdido)    TP (correto)
```

- **TN:** benignos corretamente identificados
- **FP:** benignos classificados como malignos (biópsias desnecessárias)
- **FN:** malignos classificados como benignos — minimizado pela priorização do Recall
- **TP:** malignos corretamente detectados

**Rastreamento de falsos negativos:**

Os IDs originais dos pacientes do conjunto de teste são preservados em `data/machine_learning/processed/id_test.csv`. Ao final da avaliação, os índices dos falsos negativos são cruzados com esse arquivo para identificar os IDs dos pacientes que o modelo classificou incorretamente como benignos.

```python
fn_indices = [i for i, (yt, yp) in enumerate(zip(y_test, y_pred)) if yt == 1 and yp == 0]
fn_ids = [id_test.iloc[i] for i in fn_indices]
```

---

## Modelos Avaliados

| Modelo | Classe scikit-learn | Configuração |
|---|---|---|
| Logistic Regression | `LogisticRegression` | `max_iter=10000, random_state=42` |
| Random Forest | `RandomForestClassifier` | `n_estimators=100, random_state=42` |
| Decision Tree | `DecisionTreeClassifier` | `random_state=42` |
| KNN | `KNeighborsClassifier` | `n_neighbors=5` |
| **SVM** | `SVC` | `kernel=rbf, probability=True, random_state=42` |
| Gradient Boosting | `GradientBoostingClassifier` | `random_state=42` |
| Extra Trees | `ExtraTreesClassifier` | `n_estimators=100, random_state=42` |
| MLP | `MLPClassifier` | `hidden_layer_sizes=(64,32), max_iter=1000, random_state=42` |

**Melhor modelo:** SVM (Support Vector Machine com kernel RBF)

O `random_state=42` é definido em todos os modelos e nos splits para garantir **reprodutibilidade** — qualquer execução do pipeline gera os mesmos resultados.

---

## Reprodutibilidade

Para reproduzir os resultados:

```bash
python -m src.pipeline.training_pipeline
```

Saída esperada ao final:

```
Modelo          : svm
Accuracy        : 0.9737
Recall Maligno  : 0.9524  ← métrica principal
F1 Maligno      : 0.9639
Falsos Negativos: 2       ← casos malignos não detectados
```
