import evaluate
import numpy as np
import pandas as pd
import re
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import paths_config
from model.qa_baseline import baseline_bm25
from model.qa_model import qa_model
from pipeline.s_expression_util import SExpression

RELEVANCE_TEST = True  # See and verify every answer

df = pd.read_csv(paths_config.DATASET_PATH, delimiter=';')
train_data, test_data = train_test_split(df, test_size=.1, random_state=42)

target_expressions = []
predicted_expressions = []
table_em = []
msr_em = []
dim_f1 = []
answers = []

bleu = evaluate.load("sacrebleu")
rouge = evaluate.load("rouge")

for i, row in tqdm(test_data.iterrows(), total=len(test_data), desc="Evaluating model performance",
                   bar_format='{l_bar}{bar:30}{r_bar}{bar:-30b}'):
    query = row.query
    target = row.sexp

    if query is np.nan or not query:
        continue

    try:
        target = re.sub(r'GC \w+\b', 'GC <GC>', target)
        target = re.sub(r'TC \w+\b', 'TC <TC>', target)
        sexp = SExpression.parse(target)
    except:
        print(f"Couldn't parse query {query} with target {target}")

    target_table = sexp._current_table
    target_measures = {m.value for m in sexp.obs_map[target_table].measures}
    target_dims = ({g.value for g in sexp.obs_map[target_table].dim_groups} |
                   {d.value for d in sexp.obs_map[target_table].dims})

    try:
        if RELEVANCE_TEST:
            inf_exp, answer = qa_model(query)
            if answer is not None:
                answers.append((answer, target))
        else:
            inf_exp = qa_model(query, return_sexp=True)
            # inf_exp = baseline_bm25(query, return_sexp=True)
            if inf_exp is None:
                raise Exception("Generated expression in None")
    except Exception as e:
        print(f"Couldn't generate an S-expression for {query}: {e}")
        continue

    target_expressions.append(target)
    predicted_expressions.append(str(inf_exp))

    inf_table = inf_exp._current_table
    inf_measures = {m.value for m in inf_exp.obs_map[inf_table].measures}
    inf_dims = ({g.value for g in inf_exp.obs_map[inf_table].dim_groups} |
                {d.value for d in inf_exp.obs_map[inf_table].dims})

    table_em.append(int(str(target_table) == str(inf_table)))
    msr_em.append(int(str(list(target_measures)[0]) == str(list(inf_measures)[0])))

    p = 0 if len(inf_dims) == 0 else len(inf_dims & target_dims) / len(inf_dims)
    r = 1 if len(target_dims) == 0 else len(inf_dims & target_dims) / len(target_dims)
    dim_f1.append(0 if p + r == 0 else 2 * ((p * r) / (p + r)))

rouge_score = rouge.compute(predictions=predicted_expressions, references=target_expressions,
                            rouge_types=["rouge2"])["rouge2"]
bleu_score = bleu.compute(predictions=predicted_expressions, references=target_expressions)["score"]

print(f"(Model evaluation results; {len(table_em)} samples)\n"
      f"ROUGE-2: {round(rouge_score, 3)};\n"
      f"BLEU: {round(bleu_score, 3)};\n"
      f"Table EM: {round(np.average(table_em), 3)};\n"
      f"MSR EM: {round(np.average(msr_em), 3)};\n"
      f"DIM F1: {round(np.average(dim_f1), 3)}")

if RELEVANCE_TEST:
    answer_df = []
    for i, (answer, target) in enumerate(answers):
        answer_s = str(answer.sexp)
        answer_s = re.sub(r'GC \w+\b', 'GC <GC>', answer_s)
        answer_s = re.sub(r'TC \w+\b', 'TC <TC>', answer_s)
        identical_answer = answer_s == target
        answer_df.append((int(identical_answer), f"Target expression: {target}\n{answer}"))

    df = pd.DataFrame(answer_df, columns=['relevant', 'answer'])
    df.loc[:, "answer"] = df["answer"].apply(lambda x: x.replace('\n', '\\n'))
    df.to_csv('./data/SNERT_full_answers.csv', index=False, sep='\t')

"""
Baseline (Model evaluation results; 101 samples)
          BM25+   Faiss:
ROUGE-2:   0.437;  0.374;
BLEU:     62.198; 53.025;
Table EM:  0.347;  0.396;
MSR EM:    0.198;  0.158;
DIM F1:    0.621;  0.496;

SNERTe full (Model evaluation results; 229 samples)
ROUGE-2: 0.318;
BLEU: 46.182;
Table EM: 0.188;
MSR EM: 0.1;
DIM F1: 0.398

SNERTe full/small dataset (Model evaluation results; 89 samples)
ROUGE-2: 0.361;
BLEU: 46.915;
Table EM: 0.27;
MSR EM: 0.112;
DIM F1: 0.501;

SNERTe small (Model evaluation results; 105 samples)  # 
ROUGE-2: 0.193;
BLEU: 40.278;
Table EM: 0.2;
MSR EM: 0.048;
DIM F1: 0.214

GroNLP (Model evaluation results; 105 samples)
ROUGE-2: 0.294;
BLEU: 48.039;
Table EM: 0.181;
MSR EM: 0.029;
DIM F1: 0.455

RobBERT (Model evaluation results; 105 samples)
ROUGE-2: 0.377;
BLEU: 55.042;
Table EM: 0.267;
MSR EM: 0.038;
DIM F1: 0.555
"""


""" OUT-OF-DOMAIN EVALUATION
Baseline (Model evaluation results; 219 samples)
           BM25+   Faiss:
ROUGE-2:    0.445;  0.26;
BLEU:      61.915; 40.884;
Table EM:   0.409;  0.178;
MSR EM:     0.278;  0.105;
DIM F1:     0.564;  0.358;

SNERTe small (Model evaluation results; 223 samples)
ROUGE-2: 0.147;
BLEU: 34.941;
Table EM: 0.076;
MSR EM: 0.013;
DIM F1: 0.17

GroNLP (Model evaluation results; 230 samples)
ROUGE-2: 0.185;
BLEU: 31.44;
Table EM: 0.126;
MSR EM: 0.039;
DIM F1: 0.176

RobBERT (Model evaluation results; 219 samples)
ROUGE-2: 0.217;
BLEU: 38.419;
Table EM: 0.114;
MSR EM: 0.027;
DIM F1: 0.223
"""