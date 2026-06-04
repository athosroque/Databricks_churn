<div align="center">
# Databricks ML Pipeline — Telco Churn

### Pipeline completo de Machine Learning para predição de churn em telecom

*Do dado bruto até a inferência em batch com score de risco por cliente*

![PySpark](https://img.shields.io/badge/PySpark-E25A1C?style=flat&logo=apachespark&logoColor=white)
![Databricks](https://img.shields.io/badge/Databricks_Serverless-FF3621?style=flat&logo=databricks&logoColor=white)
![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=flat&logo=mlflow&logoColor=white)
![Delta Lake](https://img.shields.io/badge/Delta_Lake-003366?style=flat&logoColor=white)
![Unity Catalog](https://img.shields.io/badge/Unity_Catalog-FF3621?style=flat&logo=databricks&logoColor=white)
![Python](https://img.shields.io/badge/Python_3.x-3776AB?style=flat&logo=python&logoColor=white)

</div>

---

<div align="center">

### 🌐 Página de Apresentação do Projeto

**[→ Ver Portfólio Completo](https://spark-churn.projetoathos.com.br)**

*Arquitetura · Stack técnica · Destaques de engenharia · Como rodar*

</div>

---

## Visão Geral

Pipeline de Machine Learning end-to-end construído para prever cancelamento (churn) de clientes de uma operadora de telecomunicações. O projeto cobre o fluxo completo de MLOps: ingestão de dados brutos, limpeza, análise exploratória, feature engineering distribuído, treinamento e comparação de modelos, registro no Model Registry e geração de predições em batch com classificação de risco por cliente.

**Dataset baixado via csv:** [Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) (Kaggle) — 7.043 clientes, 21 variáveis, ~26% de churn.

**Foi realizado o upload no Databricks como uma tabela dentro do schema.** 

---

## Arquitetura

```
CSV (Kaggle)
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   DELTA LAKE                        │
│                                                     │
│  [Bronze]       [Prata]        [Ouro]               │
│  Dado bruto ──► Limpeza  ──►  Features              │
│  sem filtro     tipagem        engineered           │
│                 regras de                           │
│                 negócio                             │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   SPARK ML PIPELINE                 │
│                                                     │
│  StringIndexer ► OneHotEncoder ► VectorAssembler    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                   MLFLOW TRACKING                   │
│                                                     │
│  LogisticRegression │ RandomForest │ GBT            │
│  AUC: 0.82    ✓      │ AUC: 0.81    │ AUC: 0.82    │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│            UNITY CATALOG MODEL REGISTRY             │
│                                                     │
│  workspace.default.telco-churn-predictor (v1)       │
└─────────────────────────────────────────────────────┘
    │
    ▼
┌─────────────────────────────────────────────────────┐
│                 BATCH INFERENCE                     │
│                                                     │
│  workspace.default.telco_predictions                │
│  score_risco: ALTO / MÉDIO / BAIXO por cliente      │
└─────────────────────────────────────────────────────┘
```

---

## Notebooks

| # | Arquivo | Responsabilidade |
|---|---------|-----------------|
| 01 | `01_ingestao_bronze.ipynb` | Lê o CSV do dataset Telco e salva na camada Bronze como tabela Delta |
| 02 | `02_limpeza_prata.ipynb` | Limpeza de nulos, tipagem de colunas, regras de negócio → camada Prata |
| 03 | `03_eda_spark_sql.ipynb` | Análise exploratória com Spark SQL: distribuições, correlações, churn por segmento |
| 04 | `04_feature_engineering.ipynb` | Cria features derivadas, monta Spark ML Pipeline, salva camada Ouro |
| 05 | `05_treinamento_mlflow.ipynb` | Treina LR, RF e GBT com rastreio MLflow; registra modelo campeão no Unity Catalog |
| 06 | `06_avaliacao_batch_inference.ipynb` | Avalia o campeão, gera predições em batch com score de risco por cliente |

---

## Features Criadas

Além das 21 colunas originais do dataset, o Notebook 04 cria três features derivadas:

| Feature | Descrição |
|---------|-----------|
| `num_services` | Contagem de serviços adicionais contratados (0–8) |
| `charges_per_tenure` | `TotalCharges / tenure` — custo médio por mês de contrato |
| `is_new_customer` | Flag `1` se `tenure ≤ 12 meses`, `0` caso contrário |

> **Atenção:** essas features devem ser recriadas manualmente antes de qualquer chamada a `pipeline_loaded.transform()` em dados novos — o pipeline Spark ML as consome, mas não as gera.

---

## Resultados dos Modelos

| Modelo | AUC-ROC | F1-Score | Status |
|--------|---------|----------|--------|
| **Logistic Regression** | **0.82** | **0.78** | **Campeão ✓** |
| Random Forest | 0.81 | 0.80 | — |
| GBT (Gradient Boosted Trees) | 0.82 | 0.81 | — |

O modelo campeão é selecionado automaticamente pelo maior AUC-ROC e registrado no Unity Catalog Model Registry como `portfolio.default.telco-churn-predictor` versão 1.

---

## Stack

| Categoria | Tecnologia |
|-----------|-----------|
| Processamento | Apache Spark (PySpark) |
| Plataforma | Databricks Serverless |
| Storage | Delta Lake — arquitetura medalhão (Bronze/Prata/Ouro) |
| Governança | Unity Catalog |
| ML | Spark ML Pipeline (StringIndexer, OneHotEncoder, VectorAssembler) |
| Experiment Tracking | MLflow + Unity Catalog Model Registry |
| Dataset | Telco Customer Churn (Kaggle) |

---

## Como Executar

### Pré-requisitos

- Workspace Databricks com Serverless habilitado
- Unity Catalog configurado com catálogo `portfolio` e schema `default`
- Dataset `WA_Fn-UseC_-Telco-Customer-Churn.csv` disponível em `/Volumes/portfolio/default/dados/`

### Configurações

```python
CATALOG = "portfolio"
SCHEMA  = "default"

# Tabelas criadas automaticamente pelos notebooks
portfolio.default.telco_bronze
portfolio.default.telco_silver
portfolio.default.telco_gold
portfolio.default.telco_predictions

# Modelo registrado
portfolio.default.telco-churn-predictor  (versão 1)

# Pipeline serializado
/Volumes/portfolio/default/modelos_ml/pipeline_model
```

### Ordem de Execução

Execute os notebooks na sequência numérica:

```
01 → 02 → 03 → 04 → 05 → 06
```

Cada notebook lê da tabela gerada pelo anterior. O Notebook 03 (EDA) pode ser pulado sem impacto no pipeline de produção.

### Restrições do Ambiente Serverless

- `spark.catalog.clearCache()` não é suportado — não utilizado
- DBFS root desabilitado — todos os paths usam `/Volumes/`
- Tabelas criadas com `saveAsTable()` em vez de `.save()` + `CREATE TABLE`
- MLflow Registry configurado com `mlflow.set_registry_uri("databricks-uc")` antes de qualquer operação com o Model Registry

---

## Estrutura do Repositório

```
databricks-ml-pipeline/
├── README.md
├── .gitignore
└── notebooks/
    ├── 01_ingestao_bronze.ipynb
    ├── 02_limpeza_prata.ipynb
    ├── 03_eda_spark_sql.ipynb
    ├── 04_feature_engineering.ipynb
    ├── 05_treinamento_mlflow.ipynb
    └── 06_avaliacao_batch_inference.ipynb
```

---

<div align="center">
  <sub>Desenvolvido por <a href="https://github.com/athosroque">Athos Roque</a> · Brasília, DF</sub>
</div>
