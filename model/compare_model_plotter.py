import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import floor

perplexity = []
eval_loss = []
train_loss = []
avg_train_loss = []

paths = [  # (model name, csv folder)
    ('SNERTe', './model/SNERTe_model/model_output'),
    ('GroNLP', './model/GroNLP_model/model_output'),
    ('RobBERT', './model/RobBERT_model/model_output'),
]

for col, func in {('loss/train', np.argmin), ('loss/eval', np.argmin), ('rouge2', np.argmax), ('bleu', np.argmax)}:
    fig = plt.figure(figsize=(8, 6))
    fig.subplots_adjust(bottom=0.2)
    ax = fig.add_subplot(1, 1, 1)

    plt_x_labels = []

    for model_name, path in paths:
        files = glob.glob(path + "/*.csv")

        df_list = (pd.read_csv(file) for file in files)
        metrics = pd.concat(df_list, ignore_index=True)

        # Remove double entries due to continuing from checkpoint
        mask = metrics.notna().all(axis=1)
        metrics = metrics[mask]
        metrics = metrics[metrics[col].ne(metrics[col].shift())]
        metrics.reset_index(inplace=True, drop=True)
        x = np.arange(metrics.shape[0])

        if len(metrics) > len(plt_x_labels):
            plt_x_labels = [f"({metrics.loc[i, 'epoch']}, {metrics.loc[i, 'step']})" if i % (floor(len(x) / 32)) == 0
                            else '' for i in x]

        best_checkpoint = metrics.iloc[func(metrics[col])]
        print(f"Best checkpoint for metric {col}, model {model_name}:"
              f"({best_checkpoint['epoch']}, {best_checkpoint['step']}); {best_checkpoint[col]}")
        ax.plot(x, metrics[col].values, label=model_name)

    print()

    ax.set_title(f"Finetuning model {col} scores")
    ax.set_xlabel('Epoch, step')

    ax.set_xticks(range(len(plt_x_labels)))
    ax.set_xticklabels(plt_x_labels, rotation=45, fontsize=8)

    ax.legend()
    plt.savefig(f"./model/model_metric_plot_{col.replace('/', '_')}.png", bbox_inches='tight')
    plt.show()
