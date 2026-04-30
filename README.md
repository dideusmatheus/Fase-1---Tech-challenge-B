# TECH CHALLENGE B

## Visão geral
Este projeto constrói a base de um sistema de suporte à decisão clínica usando aprendizado de máquina aplicado ao diagnóstico de câncer de mama. A solução reúne processamento de dados, pré-processamento, treinamento e validação de modelos para fornecer um pipeline reproduzível e orientado a métricas de segurança clínica.

## Objetivo do projeto
- Desenvolver um pipeline de Machine Learning capaz de analisar automaticamente exames médicos.
- Priorizar desempenho em recall para reduzir falsos negativos em detecção de casos malignos.
- Validar e comparar múltiplos modelos clássicos de classificação.

## Funcionalidades principais
- Carregamento e limpeza de dados
- Pré-processamento de features e target
- Treinamento de modelos de classificação
- Avaliação com métricas de classificação balanceadas
- Comparação de modelos para seleção do melhor candidato
- Execução automática via pipeline

## Tecnologias
- Python
- scikit-learn
- pandas
- NumPy
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
O dataset utilizado é o Breast Cancer Wisconsin Dataset. Os dados brutos devem estar presentes em `data/raw/data.csv`.

- Fonte: https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data/data
- Tipo: classificação binária (benigno vs maligno)
- Observações: dados sem valores nulos, com leve desbalanceamento entre classes

## Estrutura do projeto
- `data/`
  - `raw/`: dados originais
  - `processed/`: datasets preparados para treino, validação e teste
- `notebooks/`: análises exploratórias e visualizações
- `src/`: código-fonte do projeto
  - `models/`: definição de modelos e módulos de avaliação
  - `pipeline/`: fluxo de treinamento e validação

## Fluxo do pipeline
1. Carregamento dos dados
2. Limpeza e pré-processamento
   - remoção da coluna `id`
   - codificação do target
   - remoção de duplicatas
3. Treinamento de múltiplos modelos
4. Avaliação em conjunto de validação
5. Teste final e relatório de resultados

## Modelos avaliados
- Logistic Regression
- Random Forest
- Decision Tree
- KNN
- SVM
- Gradient Boosting
- Extra Trees
- MLP

## Métricas de avaliação
- Accuracy
- Precision
- Recall
- F1-score

> O foco principal do projeto é **recall**, para minimizar o número de falsos negativos em casos malignos.

## Resultados principais
- Melhor modelo: `Logistic Regression`
- Accuracy: `0.9825`
- Recall (classe maligno): `0.9762`
- F1-score (classe maligno): `0.9762`
- Falsos negativos: `1`
<img width="453" height="190" alt="image" src="https://github.com/user-attachments/assets/257cdec8-a226-48be-b724-6101b7869b2d" />

