import glob
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from math import ceil

perplexity = []
eval_loss = []
train_loss = []
avg_train_loss = []

path = './model/SNERTe_model/model_output'
files = glob.glob(path + "/*.csv")

df_list = (pd.read_csv(file) for file in files)
metrics = pd.concat(df_list, ignore_index=True)

# Remove double entries due to continuing from checkpoint
mask = metrics.iloc[metrics['loss/eval'].first_valid_index():].notna().all(axis=1)
metrics = metrics[pd.concat([pd.Series([True] * metrics['loss/eval'].first_valid_index()), mask])]
x = np.arange(metrics.shape[0])

fig = plt.figure(figsize=(8, 6))
fig.subplots_adjust(bottom=0.2)
ax = fig.add_subplot(1, 1, 1)

# for col, func in {('loss/train', np.argmin)}:
for col, func in {('loss/eval', np.argmin), ('rouge2', np.argmax)}:
    best_checkpoint = metrics.iloc[func(metrics[col])]
    print(f"Best checkpoint for metric {col}: ({best_checkpoint['epoch']}, {best_checkpoint['step']}); "
          f"{best_checkpoint[col]}")
    ax.plot(x, metrics[col].values, label=col)

ax.set_title('Fine-tuning/training model')
ax.set_xlabel('Epoch, step')

ax.set_xticks(range(len(x)))
ax.set_xticklabels([f"({metrics.loc[i, 'epoch']}, {metrics.loc[i, 'step']})" if i % (ceil(len(x) / 32)) == 0
                    else '' for i in x],
                   rotation=45, fontsize=8)

ax.legend()
plt.show()
