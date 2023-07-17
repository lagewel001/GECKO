import hdbscan
import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
import umap
import umap.plot
from sentence_transformers import SentenceTransformer
from networkx.algorithms.components import connected_components
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from tqdm import tqdm
from transformers import RobertaTokenizerFast, RobertaModel

import paths_config

dataset = pd.read_csv(paths_config.DATASET_PATH, on_bad_lines='skip', delimiter=';')
dataset = dataset.dropna(subset=['query'])
dataset['table_id'] = ''

tokenizer = RobertaTokenizerFast.from_pretrained(paths_config.TOKENIZER_PATH)
model = RobertaModel.from_pretrained(paths_config.START_MODEL_PATH)

# model = SentenceTransformer(paths_config.SENT_TRANSFORMER_PATH)

embedding_df = pd.DataFrame(index=dataset.index, columns=range(768))
for idx, row in tqdm(dataset.iterrows(), total=dataset.shape[0],
                     bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
    table_id = row['sexp'].split('(VALUE (')[1].split(' ')[0]
    dataset.loc[idx, 'table_id'] = table_id

    query = row['query']
    if query is not np.nan and query:
        input_ids = tokenizer(row['query'],
                              padding=True,
                              truncation=True,
                              return_tensors='pt',
                              add_special_tokens=True)
        with torch.no_grad():
            output = model(**input_ids, output_hidden_states=True)
            cls = output.last_hidden_state[:, 0, :]
            embedding_df.loc[idx, :] = cls.cpu().detach()

        # embedding = model.encode(query)
        # embedding_df.loc[idx, :] = embedding

with open('./data/table_themes.json', 'r') as file:
    table_themes = json.loads(file.read())
    dataset['group'] = dataset['table_id'].map(table_themes)

# UMAP for visualisation
reducer = umap.UMAP(n_components=2, n_neighbors=10, min_dist=0.0, random_state=777)
projected_embeddings = reducer.fit_transform([embedding_df.iloc[idx].values for idx in
                                              range(dataset.shape[0]) if dataset.iloc[idx].group != -1])

# MATPLOTLIB
fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

labels = dataset.group
cmap = plt.get_cmap('tab20')
for l, c in zip(labels.unique(), cmap.colors):
    ax.scatter(projected_embeddings[labels == l, 0],
               projected_embeddings[labels == l, 1],
               label=l, color=c)

ax.text(0.99, 0.01, f"UMAP: n_neighbors={reducer.n_neighbors}, min_dist={reducer.min_dist}",
        verticalalignment='bottom',
        horizontalalignment='right',
        transform=ax.transAxes,
        fontsize=12)
plt.legend()
plt.title('UMAP of SentenceTransformer token embeddings of training questions coloured on table theme.')
plt.show()

# UMAP for clustering
clusterable_reducer = umap.UMAP(
    n_neighbors=20,
    min_dist=0.0,
    n_components=50,
    random_state=777,
)

clusterable_embedding = clusterable_reducer.fit_transform([embedding_df.iloc[idx].values for idx in
                                                           range(dataset.shape[0]) if dataset.iloc[idx].group != -1])
hdbscan_labels = hdbscan.HDBSCAN(
    min_samples=1,
    min_cluster_size=10,
).fit_predict(clusterable_embedding)

fig = plt.figure(figsize=(10, 10))
ax = fig.add_subplot()

clustered = (hdbscan_labels >= 0)
ax.scatter(projected_embeddings[~clustered, 0],
           projected_embeddings[~clustered, 1],
           color=(0.5, 0.5, 0.5),
           alpha=0.5)

ax.scatter(projected_embeddings[clustered, 0],
           projected_embeddings[clustered, 1],
           c=hdbscan_labels[clustered],
           cmap=cmap)
plt.show()

print(f"Adj. rand score: {adjusted_rand_score(labels[clustered], hdbscan_labels[clustered])}")
print(f"Adj. mutual info score: {adjusted_mutual_info_score(labels[clustered], hdbscan_labels[clustered])}")
print(f"Clustered data percentage: {np.sum(clustered) / len(dataset)}")
