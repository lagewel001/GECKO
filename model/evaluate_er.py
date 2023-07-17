import json
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import paths_config
from pipeline.entity_recognition import EntityRecognizer
from pipeline.logical_forms import uri_to_code, TC, GC, MSR, DIM
from pipeline.s_expression_util import SExpression

er = EntityRecognizer()

df = pd.read_csv(paths_config.DATASET_PATH, delimiter=';')
train_data, test_data = train_test_split(df, test_size=.1, random_state=42)

with open(paths_config.FAISS_URI_MAP_PATH, 'r') as file:
    uri_map = json.load(file)
    codes = [uri_to_code(uri) for uri in uri_map.values()]

table_acc = []
table_p = []
table_mrr = []

msr_acc = []
msr_p = []

dim_p = []
dim_r = []
dim_f1 = []

ranking_lengths = []

kwargs = {
    'faiss_k': 100,
    'table_cutoff': 7,
    'obs_cutoff': 10
}
er_func = er.get_candidate_nodes_embeddings

# kwargs = {
#     'k': 25,
#     'table_cutoff': 5,
#     'obs_cutoff': 10
# }
# er_func = er.get_candidate_nodes_bm25_ranked()


test_data.dropna(inplace=True)
for i, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Calculating ER-step prompt scores",
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
    query = row.query
    target = row.sexp

    try:
        target = re.sub(r'GC \w+\b', 'GC <GC>', target)
        target = re.sub(r'TC \w+\b', 'TC <TC>', target)
        sexp = SExpression.parse(target)
    except:
        print(f"Couldn't parse query {query} with target {target}")

    target_table = str(sexp._current_table)
    target_measures = {m.value for m in sexp.obs_map[sexp._current_table].measures}
    target_groups = {g.value for g in sexp.obs_map[sexp._current_table].dim_groups
                     if not (isinstance(g, TC) or isinstance(g, GC))}
    target_dims = {d.value for d in sexp.obs_map[sexp._current_table].dims
                   if d.value not in ['<TC>', '<GC>']}

    ranked_nodes, g = er_func(query, **kwargs)

    table_ranking = []
    msr_ranking = []
    dim_ranking = []
    for table, (_, obs) in ranked_nodes.items():
        table_ranking.append(uri_to_code(table))
        for node in obs:
            if node in MSR.rdf_ns:
                msr_ranking.append(uri_to_code(node))
            elif node in DIM.rdf_ns:
                dim_ranking.append(uri_to_code(node))
            else:
                raise Exception("Wot happened?")

    table_acc.append(int(str(target_table) in table_ranking))
    table_p.append(0 if len(table_ranking) == 0 else len(set(table_ranking) & {target_table}) / len(table_ranking))
    table_mrr.append(1 / (table_ranking.index(target_table) + 1) if target_table in table_ranking else .0)

    msr_acc.append(sum([1 if msr in msr_ranking else 0 for msr in target_measures]) / len(target_measures))
    msr_p.append(0 if len(msr_ranking) == 0 else len(set(msr_ranking) & target_measures) / len(set(msr_ranking)))

    target = target_groups | target_dims
    p = 0 if len(dim_ranking) == 0 else len(set(dim_ranking) & target) / len(set(dim_ranking))
    r = 1 if len(target) == 0 else len(set(dim_ranking) & target) / len(target)
    dim_p.append(p)
    dim_r.append(r)
    dim_f1.append(0 if p + r == 0 else 2 * ((p * r) / (p + r)))

    ranking_lengths.append(len(table_ranking) + len(msr_ranking) + len(dim_ranking))

print(f"({kwargs}, {len(ranking_lengths)} samples)\n"
      f"Table acc: {round(np.average(table_acc), 4)};\n"
      f"Table P: {round(np.average(table_p), 4)};\n"
      f"Table MRR: {round(np.average(table_mrr), 4)};\n"
      f"MSR acc: {round(np.average(msr_acc), 4)};\n"
      f"MSR P: {round(np.average(msr_p), 4)};\n"
      f"DIM P: {round(np.average(dim_p), 4)};\n"
      f"DIM R: {round(np.average(dim_r), 4)};\n"
      f"DIM F1: {round(np.average(dim_f1), 4)};\n"
      f"Average number of nodes in prompt: {round(np.average(ranking_lengths), 4)}")
