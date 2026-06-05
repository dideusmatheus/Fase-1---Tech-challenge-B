# TECH CHALLENGE B

## Visão geral
Este projeto constrói a base de um sistema de suporte à decisão clínica usando aprendizado de máquina aplicado ao diagnóstico de câncer de mama. A solução reúne processamento de dados, pré-processamento, treinamento e validação de modelos para fornecer um pipeline reproduzível e orientado a métricas de segurança clínica.

## Objetivo do projeto
- Desenvolver um pipeline de Machine Learning capaz de analisar automaticamente exames médicos.
- Priorizar desempenho em recall para reduzir falsos negativos em detecção de casos malignos.
- Validar e comparar múltiplos modelos clássicos de classificação.

## Funcionalidades principais
- Carregamento e limpeza de dados
- Pré-processamento de features e target com normalização (StandardScaler)
- Treinamento de modelos de classificação com persistência em `.pkl`
- Avaliação com métricas de classificação balanceadas e AUC-ROC
- Comparação de modelos com critério primário em recall (classe maligno)
- Identificação dos IDs de pacientes com falsos negativos no conjunto de teste
- Execução automática via pipeline

## Tecnologias
- Python
- scikit-learn
- pandas
- NumPy
- Matplotlib
- Seaborn
- joblib
- Jupyter Notebook (análises exploratórias)

## Como executar
1. Clone o repositório:
```bash
git clone <URL_DO_REPOSITORIO>
cd "PROJETO Fase 1 - Tech challenge B"
```
2. Crie e ative um ambiente virtual:
```bash
python -m venv .venv
.venv\Scripts\activate
```
3. Instale as dependências:
```bash
pip install -r requirements.txt
```
4. Execute o pipeline completo:
```bash
python -m src.pipeline.training_pipeline
```

## Dados
O dataset utilizado é o Breast Cancer Wisconsin Dataset. Os dados brutos devem estar presentes em `data/machine_learning/raw/data.csv`.

- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- Tipo: classificação binária (benigno vs maligno)
- Observações: 569 amostras, 30 features numéricas, leve desbalanceamento entre classes (~63% benigno, ~37% maligno)

## Estrutura do projeto
- `data/`
  - `machine_learning/`
    - `raw/`: dados originais
    - `processed/`: datasets preparados para treino, validação e teste
- `models/`
  - `machine_learning/`: modelos treinados salvos em formato `.pkl`
- `notebooks/`: análises exploratórias e visualizações (`eda.ipynb`)
- `src/`: código-fonte do projeto
  - `machine_learning/`
    - `data/`: carregamento e pré-processamento dos dados
    - `training.py`: treinamento dos modelos
    - `validation.py`: seleção do melhor modelo
    - `test.py`: avaliação final no conjunto de teste
  - `pipeline/`: fluxo de treinamento e validação (`training_pipeline.py`)

## Fluxo do pipeline
1. Carregamento dos dados
2. Limpeza e pré-processamento
   - Remoção da coluna `id`
   - Remoção de colunas com mais de 40% de valores nulos
   - Remoção de duplicatas
   - Codificação do target (M→1, B→0)
   - Divisão estratificada: 64% treino / 16% validação / 20% teste
   - Normalização com `StandardScaler` (ajustado apenas no treino)
3. Treinamento de múltiplos modelos (salvos em `.pkl`)
4. Avaliação e seleção do melhor modelo no conjunto de validação
5. Teste final e relatório de resultados com identificação de falsos negativos

## Modelos avaliados
- Logistic Regression
- Random Forest
- Decision Tree
- KNN
- SVM
- Gradient Boosting
- Extra Trees
- MLP

## Critério de seleção de modelo
A seleção do melhor modelo segue um sistema de ranking em três níveis:
1. **Critério primário:** Recall da classe maligno (minimizar falsos negativos)
2. **Desempate 1:** Precision da classe maligno (minimizar falsos positivos)
3. **Desempate 2:** F1-score da classe maligno

## Métricas de avaliação
- Accuracy
- Precision
- Recall
- F1-score
- AUC-ROC

> O foco principal do projeto é **recall**, para minimizar o número de falsos negativos em casos malignos.

## Resultados principais
- Melhor modelo: `SVM`
- Accuracy: `0.9737`
- Recall (classe maligno): `0.9524`
- F1-score (classe maligno): `0.9639`
- Falsos negativos: `2`

<img width="452" height="198" alt="image" src="https://github.com/user-attachments/assets/18af7ba4-6917-4556-8866-d965b9f8dd36" />
