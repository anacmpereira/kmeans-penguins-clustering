# TAREFA MÓDULO 30 - KMEANS

# 1. Importando bibliotecas
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

# 2. Importando base de dados
pd.set_option('display.max_columns', None)
df = sns.load_dataset('penguins')
print(df)

# 1) Nesse exercício você deve verificar se temos variáveis missing, faltantes e excluir esses valores do dataset, também
# deve excluir as colunas com valores categóricos, que não utilizaremos para o k-means

# Verificando e tratando dados ausentes
print((df.isnull().sum()/len(df)) * 100)

# Existem valores ausentes em quase todas as colunas, mas a quantidade de valores ausentes é pequena quando comparamos
# com toda a base de dados, por isso vou excluí-los

df.dropna(inplace=True)
print(df.isnull().sum())

# Excluindo colunas com valores categóricos
print(df.info())
print(df.head()) # as variáveis categóricas são 'species', 'island' e 'sex'
df.drop(columns=['species', 'island', 'sex'], inplace=True)
print("Primeiras linhas do dataframe: ")
print(df.head())

# 2) Visualize a análise descritiva dos seus dados utilizando a função vista em aula. É possível identificar possíveis
# agrupamentos? Se sim, quantos?
sns.pairplot(df)
plt.show()

# A análise exploratória por meio do pairplot permitiu visualizar a distribuição das quatro variáveis e suas relações
# par a par. Em algumas combinações foi possível observar concentrações de pontos que sugerem possíveis agrupamentos, como
# 'bill_depth_mm' vs. 'bill_length_mm'; 'bill_depth_mm' vs. 'flipper_length_mm'; 'bill_depth_mm' vs. 'body_mass_g'.
# Em algumas relações podemos observar a formação de aprox. três concentrações, indicando a existência de possíveis agrupamentos
# naturais

# 'bill_depth_mm' vs. 'bill_depth_mm
fig = px.scatter(df, x='bill_length_mm', y='bill_depth_mm',
                 title='Gráfico de dispersão: Comprimento vs. Profundidade do bico',
                 labels={'bill_length_mm': 'Comprimento do bico', 'bill_depth_mm': 'Profundidade do bico'})
fig.show() # mostra três possíveis agrupamentos

# 'bill_depth_mm' vs. 'flipper_length_mm'
fig = px.scatter(df, x='bill_length_mm', y='flipper_length_mm',
                 title='Gráfico de dispersão: Comprimento do bico vs. Comprimento da barbatana',
                 labels={'bill_length_mm': 'Comprimento do bico', 'flipper_length_mm': 'Comprimento da barbatana'})
fig.show() # mostra três possíveis agrupamentos

# 'bill_depth_mm' vs. 'body_mass_g'
fig = px.scatter(df, x='bill_length_mm', y='body_mass_g',
                 title='Gráfico de dispersão: Comprimento do bico vs. Massa corporal',
                 labels={'bill_length_mm': 'Comprimento do bico', 'body_mass_g': 'Massa corporal'})
fig.show() # mostra três possíveis agrupamentos

# 'flipper_length_mm' vs. 'body_mass_g'
fig = px.scatter(df, x='flipper_length_mm', y='body_mass_g',
                 title='Gráfico de dispersão: Comprimento da barbatana vs. Massa corporal',
                 labels={'flipper_length_mm': 'Comprimento da barbatana', 'body_mass_g': 'Massa corporal'})
fig.show() # mostra dois possíveis agrupamentos

# 3) Realize a padronização e visualize os dados
df_padronizado = df.copy() # faz uma cópia dos dados
colunas_padronizar = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']

# Inicializar o StandardScaler
scaler = StandardScaler()

# Ajustar e transformar os dados selecionados
df_padronizado[colunas_padronizar] = scaler.fit_transform(df[colunas_padronizar])
print(df_padronizado)

# 4) Aplique o algoritmo k-means escolhendo o número de clusters.
kmeans_penguins = KMeans(n_clusters=3, n_init=10, random_state=42)
kmeans_penguins.fit(df_padronizado)

centroides = kmeans_penguins.cluster_centers_
print("Centróides: ")
print(centroides)

centroides_padronizados = kmeans_penguins.cluster_centers_

centroides_originais = scaler.inverse_transform(centroides_padronizados) # "DESPADRONIZAR" OS CENTROIDES
print("Centróides originais: ")
print(centroides_originais)

# Interpretando dos clusters
# O primeiro grupo apresentou indivíduos com menor comprimento de bico, menor comprimento de nadadeira e menor
# massa corporal. O segundo agrupou indivíduos de maior porte, caracterizados por maiores comprimentos de nadadeira e
# maior massa corporal, além de menor profundidade do bico. Já o terceiro cluster apresentou comprimento de bico
# semelhante ao segundo, porém com maior profundidade do bico e valores intermediários de comprimento de nadadeira
# e massa corporal

# 5) Construa duas matrizes de dispersão como realizadas em aula indicando os pontos e centróides

# Adicionando os rótulos dos clusters ao DataFrame
labels = kmeans_penguins.labels_
print(labels)

# Criando um df com os dados originais e rótulos de clusters
df_clusters = pd.DataFrame({
    'bill_length_mm': df['bill_length_mm'],
    'bill_depth_mm': df['bill_depth_mm'],
    'flipper_length_mm': df['flipper_length_mm'],
    'body_mass_g': df['body_mass_g'],
    'cluster': labels.astype(str)
})

# a) 'bill_length_mm' vs. 'bill_depth_mm'
fig = px.scatter(
    df_clusters,
    x='bill_length_mm',
    y='bill_depth_mm',
    color='cluster',
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7,
    title='Clusters de agrupamento de pinguins'
)

fig.add_scatter(
    x=centroides_originais[:, 0],
    y=centroides_originais[:, 1],
    mode='markers',
    marker=dict(color='black', symbol='x', size=16),
    name='Centroides'
)

fig.update_layout(xaxis_title='Comprimento do bico (mm)',
                  yaxis_title='Profundidade do bico)',
                  legend_title='Clusters')

fig.show()

# b) 'flipper_length_mm' vs. 'body_mass_g'
fig = px.scatter(
    df_clusters,
    x='flipper_length_mm',
    y='body_mass_g',
    color='cluster',
    color_discrete_sequence=px.colors.qualitative.Set1,
    opacity=0.7,
    title='Clusters de agrupamento de pinguins'
)

fig.add_scatter(
    x=centroides_originais[:, 2],
    y=centroides_originais[:, 3],
    mode='markers',
    marker=dict(color='black', symbol='x', size=16),
    name='Centroides'
)

fig.update_layout(xaxis_title='Comprimento da nadadeira (mm)', yaxis_title='Massa corporal (g)',
                  legend_title='Cluster')

fig.show()

