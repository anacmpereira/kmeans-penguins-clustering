# kmeans-penguins-clustering
Aplicação do algoritmo K-Means na base Palmer Penguins, incluindo análise exploratória, tratamento dos dados, padronização das variáveis e visualização dos clusters.
# Clusterização com K-Means na Base Palmer Penguins

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-KMeans-orange?logo=scikitlearn)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas)
![Seaborn](https://img.shields.io/badge/Seaborn-Visualization-4C72B0)
![Plotly](https://img.shields.io/badge/Plotly-Interactive%20Graphs-3F4F75?logo=plotly)

## Sobre o projeto

Este projeto apresenta a aplicação do algoritmo **K-Means**, uma técnica de **aprendizado de máquina não supervisionado**, utilizando a base de dados **Palmer Penguins**, disponível na biblioteca Seaborn.

O objetivo é identificar agrupamentos naturais entre os indivíduos a partir de características morfológicas, realizando desde o pré-processamento dos dados até a interpretação dos clusters encontrados.

---

## Base de dados

A base **Palmer Penguins** contém informações morfológicas de diferentes espécies de pinguins, incluindo:

- Comprimento do bico
- Profundidade do bico
- Comprimento da nadadeira
- Massa corporal

As variáveis categóricas foram removidas para que apenas atributos numéricos fossem utilizados pelo algoritmo.

---

## Etapas desenvolvidas

O projeto contempla as seguintes etapas:

- Verificação de valores ausentes
- Remoção de registros com valores faltantes
- Exclusão das variáveis categóricas
- Análise exploratória utilizando **Pairplot**
- Visualização de possíveis agrupamentos
- Padronização dos dados com **StandardScaler**
- Aplicação do algoritmo **K-Means**
- Cálculo dos centróides
- Conversão dos centróides para a escala original
- Visualização gráfica dos clusters obtidos

---

## Tecnologias utilizadas

- Python
- Pandas
- Seaborn
- Plotly
- Scikit-Learn
- Matplotlib

---

## Resultados

A análise exploratória indicou a existência de aproximadamente **três agrupamentos naturais** nos dados.

Após a aplicação do algoritmo **K-Means** com **k = 3**, os clusters encontrados apresentaram características distintas em relação às medidas morfológicas dos pinguins, especialmente quanto ao comprimento do bico, comprimento da nadadeira e massa corporal.

Os resultados foram visualizados por meio de gráficos de dispersão contendo:

- os indivíduos coloridos por cluster;
- os centróides representados por marcadores em formato de "X".

---

## Estrutura do projeto

```
.
├── tarefa_k-means.py
└── README.md
```

---

## Como executar

Clone o repositório:

```bash
git clone https://github.com/seu-usuario/kmeans-penguins-clustering.git
```

Entre na pasta:

```bash
cd kmeans-penguins-clustering
```

Instale as dependências:

```bash
pip install pandas seaborn matplotlib plotly scikit-learn
```

Execute:

```bash
python tarefa_k-means.py
```

---

## Conceitos abordados

- Aprendizado não supervisionado
- Clusterização
- K-Means
- Padronização de dados
- Análise exploratória
- Visualização de dados
- Centróides
- Machine Learning

---

## Autora

**Ana Carolina Martins Pereira**

Projeto desenvolvido como atividade prática do curso de Cientista de Dados da EBAC .
