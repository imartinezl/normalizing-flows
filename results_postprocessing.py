# %%

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# %% 2D

datasets = ["MOONS", "CIRCLES", "CRESCENT", "CRESCENTCUBED", "SINEWAVE", "ABS", "SIGN", "TWOSPIRALS", "CHECKERBOARD", "FOURCIRCLES", "DIAMOND"]

folder = "results2D"

df = pd.read_csv("results2D.csv")

from pathlib import Path
path = "results2D-summary"
Path(path).mkdir(parents=True, exist_ok=True)

# %%
df.groupby("dataset")["test_loss"].idxmin()

import os

a = df.iloc[df.groupby("dataset")["test_loss"].idxmin()]
for i, row in a.iterrows():
    p = os.path.join(row["folder"], row["dataset"], str(row["run"]) )
    from distutils.dir_util import copy_tree
    copy_tree(p, os.path.join(path, row["dataset"]))
    

# %%
col = "dataset"
x = "tess_size"
y = "test_loss"
hue = "flow_steps"

# col = "dataset"
# y = "test_loss"
# x = "flow_steps"
# hue = "hidden_layers"

df.groupby([col, x, hue])[y].min()


df_plot = df.groupby([col, x, hue])[y].min().reset_index()
g = sns.FacetGrid(df_plot, col=col, col_wrap=4, sharex=True, sharey=False, height=3, aspect=1, despine=False)
g.map_dataframe(sns.lineplot, x=x, y=y, hue=hue, palette="viridis")
g.add_legend()
g.map_dataframe(sns.scatterplot, x=x, y=y, hue=hue)


# %%

col = "dataset"
df_plot = pd.melt(df, 
    id_vars=[col],
    value_vars=["train_loss", "test_loss"])

x = "variable"
y = "value"

g = sns.FacetGrid(df_plot, col=col, col_wrap=4, sharex=True, sharey=False, height=3, aspect=1, despine=False)
g.map_dataframe(sns.boxplot, x=x, y=y, palette="viridis")

# %%
col = "dataset"
df_plot = pd.melt(df, 
    id_vars=[col],
    value_vars=["train_logprob_mean", "test_logprob_mean"])

x = "variable"
y = "value"

g = sns.FacetGrid(df_plot, col=col, col_wrap=4, sharex=True, sharey=False, height=3, aspect=1, despine=False)
g.map_dataframe(sns.boxplot, x=x, y=y, palette="viridis")


# %%

def pad(x,d):
    return f"{x:.{d}f}"
def my_round(mean, std, dec, sep):
    vfunc = np.vectorize(pad)
    return vfunc(mean, dec).astype(object) + sep + vfunc(std, dec).astype(object)

# %% RESULTS 1D

train_size = 5000
test_size = 2000
batch_size = 256
tessellation_size = [4,8,16,32]
flow_steps = [1,2,3]
epochs = 500
lr = [0.0001, 0.001, 0.005]
df = pd.read_csv("results1D.csv")
df.apply(np.unique)

datasets = pd.DataFrame(dict(
    dataset = ['UNIFORM', 'GAUSSIANMIXTURE', 'GAUSSIAN', 'POWER', 'BLOBS'],
    dimension = 1,
    train_points = 5000,
    test_points = 2000
))

# %%

summary = df.iloc[df.groupby("dataset")["test_logprob_mean"].idxmax()]
summary["loss"] = my_round(summary["train_loss"], summary["test_loss"], 2, " / ")
summary["logprob"] = my_round(summary["train_logprob_mean"], summary["train_logprob_std"], 2, " $\pm$ ") + " / " + my_round(summary["test_logprob_mean"], summary["test_logprob_std"], 2, " $\pm$ ")

summary.to_latex("results1D.tex", float_format="%.2f", index=False, escape=False,
    columns=["dataset", "tess_size", "flow_steps", "loss", "logprob"],
    header =["Dataset", "Tessellation Size", "Flow Steps", "Loss (train/test)", "$\log p(x)$"])

# %%
summary["logprob"] = my_round(summary["test_logprob_mean"], summary["test_logprob_std"], 2, " $\pm$ ")
summary.to_latex(
    "results1D_test.tex", float_format="%.2f", index=False, escape=False,
    columns = ["dataset", "tess_size", "flow_steps", "logprob"],
    header =["Dataset",  "Tessellation Size", "Flow Steps", "$\log p(\mathbf{x})$"]
)

# %%
summary["lr"] = summary["lr"].astype(str)
summary_t = summary[["dataset", "train_size", "test_size", "batch_size", 
    "tess_size", "flow_steps", "epochs", "lr", "logprob"]].drop_duplicates().set_index("dataset").transpose()
summary_t.index = ["Train Size", "Test Size", "Batch Size", 
    "Tessellation Size", "Flow Steps", "Epochs", "Learning Rate", "$\log p(\mathbf{x})$"]
    
summary_t.to_latex(buf="results1D_test_t.tex", index=True, escape=False, bold_rows=False)




# %% RESULTS 2D

train_size = 5000
test_size = 2000
batch_size = 256
hidden_dim = [8,16]
hidden_layers = [1,2,4]
tessellation_size = [4,8,16,32]
flow_steps = [1,2,3,4,8]
epochs = 500
lr = 0.001
df = pd.read_csv("results2D.csv")
df.apply(np.unique)

datasets = pd.DataFrame(dict(
    dataset = ['MOONS', 'CIRCLES', 'CRESCENT', 'CRESCENTCUBED', 'SINEWAVE', 'ABS',
       'SIGN', 'TWOSPIRALS', 'CHECKERBOARD', 'FOURCIRCLES', 'DIAMOND'],
    dimension = 2,
    train_points = 5000,
    test_points = 2000
))

# %%

summary = df.iloc[df.groupby("dataset")["test_logprob_mean"].idxmax()]
summary["loss"] = my_round(summary["train_loss"], summary["test_loss"], 2, " / ")
summary["logprob"] = my_round(summary["train_logprob_mean"], summary["train_logprob_std"], 2, " $\pm$ ") + " / " + my_round(summary["test_logprob_mean"], summary["test_logprob_std"], 2, " $\pm$ ")



summary.to_latex("results2D.tex", float_format="%.2f", index=False, escape=False,
    columns=["dataset", "hidden_dim", "hidden_layers", "tess_size", "flow_steps", "loss", "logprob"],
    header =["Dataset", "\# Neurons per Layer", "\# Hidden Layers" , "Tessellation Size", "Flow Steps", "Loss (train/test)", "$\log p(x)$"])

# %%
summary["logprob"] = my_round(summary["test_logprob_mean"], summary["test_logprob_std"], 2, " $\pm$ ")
summary.to_latex(
    "results2D_test.tex", float_format="%.2f", index=False, escape=False,
    columns = ["dataset", "hidden_dim", "hidden_layers", "tess_size", "flow_steps", "logprob"],
    header =["Dataset", "\# Neurons per Layer", "\# Hidden Layers" , "Tessellation Size", "Flow Steps", "$\log p(\mathbf{x})$"]
)

# %%
summary["lr"] = summary["lr"].astype(str)
summary_t = summary[["dataset", "train_size", "test_size", "batch_size", 
    "hidden_dim", "hidden_layers", "tess_size", "flow_steps", 
    "epochs", "lr", "logprob"]].drop_duplicates().set_index("dataset").transpose()
summary_t.index = ["Train Size", "Test Size", "Batch Size", 
    "\# Neurons per Layer", "\# Hidden Layers" , "Tessellation Size", "Flow Steps", 
    "Epochs", "Learning Rate", "$\log p(\mathbf{x})$"]
    
summary_t.to_latex(buf="results2D_test_t.tex", index=True, escape=False, bold_rows=False)


# %% RESULTS ND
# train_size = [10000, 30000]
# test_size = [2000, 3000, 5000]
batch_size = 512
hidden_dim = [64]
hidden_layers = [4]
tessellation_size = [5, 10, 20, 50]
flow_steps = [3,5,8,10,15,20]
epochs = [128,200,250,256,325,350,850]
lr = [0.0003,0.0005]
df = pd.read_csv("resultsND.csv")
df_others = pd.read_csv("resultsND_others.csv")

# %%


datasets = pd.DataFrame(dict(
    dataset = ['SWISSROLL', 'SCURVE', 'POWER', 'GAS', 'HEPMASS', 'MINIBOONE', 'BSDS300'],
    dimension = [3,3,6,8,21,43,63],
    train_points = [30000, 30000, 1615917, 852174, 315123, 29556, 1000000],
))
df = pd.merge(df, datasets, on="dataset")
df_others = pd.merge(df_others, datasets, on="dataset")

df["model_type"] = "DIFW (" + df["model_type"] + ")"
df = df[df.allowed != "NO"]
df = df.reset_index(drop=True)
# %%

summary = df.iloc[df.groupby(["dataset","model_type"])["test_logprob_mean"].idxmax()]

summary["logprob"] = my_round(summary["test_logprob_mean"], np.maximum(0.01, summary["test_logprob_std"]), 2, " $\pm$ ")
summary["lr"] = summary["lr"].astype(str)
summary.to_latex("resultsND.tex", float_format="%.2f", index=False, escape=False,
    columns=["model_type", "dataset", "dimension", "train_points", "batch_size", "hidden_dim", "hidden_layers", "tess_size", "flow_steps", "epochs", "lr", "logprob"],
    header=["Model", "Dataset", "Dimension", "Train Size", "Batch Size", "\# Neurons per Layer", "\# Hidden Layers", "Tessellation Size", "Flow Steps", "Epochs", "Learning Rate", "$\log p(\mathbf{x})$"])

# %%

summary = df.iloc[df.groupby(["dataset"])["test_logprob_mean"].idxmax()]
summary["lr"] = summary["lr"].astype(str)
summary_t = summary[["dataset", "train_points", "dimension", "batch_size", 
    "hidden_dim", "hidden_layers", "tess_size", "flow_steps", 
    "epochs", "lr"]].drop_duplicates().set_index("dataset").transpose()
summary_t.index = ["Train Points", "Dimension", "Batch Size", 
    "\# Neurons per Layer", "\# Hidden Layers" , "Tessellation Size", "Flow Steps", 
    "Epochs", "Learning Rate"]
    
summary_t.to_latex(buf="resultsND_parameters.tex", index=True, escape=False, bold_rows=False)

# %%
summary = df.iloc[df.groupby(["dataset","model_type"])["test_logprob_mean"].idxmax()]
df_all = pd.concat([summary, df_others])[["dataset", "model_type", "train_points", "test_logprob_mean", "test_logprob_std"]].reset_index(drop=True)
df_all["logprob"] = my_round(df_all["test_logprob_mean"], np.maximum(0.01, df_all["test_logprob_std"]), 2, " $\pm$ ")

df_table = df_all.pivot(index="model_type", columns="dataset", values="logprob")
df_table.reindex(['DIFW (AR)', 'DIFW (CL)', 'BLOCK-NAF', 'FFJORD', 'GLOW', 'MAF', 'NAF',
       'Q-NSF (AR) ', 'Q-NSF (C)', 'RQ-NSF (AR) ', 'RQ-NSF (C) ', 'SOS']).drop(["SCURVE", "SWISSROLL"], axis=1).to_latex(buf="resultsND_comparison.tex", index=True, escape=False, bold_rows=False)

# %% PLOT DIFFERENCE PERCENTAGE
df_plot = df_all[["model_type", "dataset", "test_logprob_mean"]][~df_all.dataset.isin(["SCURVE", "SWISSROLL"])]

fig, axs = plt.subplots(1,5, figsize=(12/1.3,4/1.3), gridspec_kw=dict(width_ratios=[1,1,1,1,1]))
plt.subplots_adjust(wspace=1)
for k, dataset in enumerate(df_plot.dataset.unique()):
    print(k, dataset)
    data = df_plot[df_plot.dataset == dataset]
    sns.swarmplot(data=data, y="test_logprob_mean", color="gray", orient="v", ax=axs[k], zorder=-1)
    axs[k].boxplot(x=data["test_logprob_mean"], widths=0.4,showfliers=False, positions=[-0.5])
    
    for model, color in zip(["DIFW (AR)", "DIFW (CL)"], ["#D90368", "#00CC66"]):
        x = data[data.model_type == model]["test_logprob_mean"]
        axs[k].scatter(x=0, y=x, color=color)
        axs[k].text(0.1, x, model.replace(" ", "\n"), color=color, va="center")

    
    axs[k].set_xlim(-1,0.5)
    axs[k].set_title(dataset, x=0, y=-0.2)

    axs[k].spines['top'].set_visible(False)
    axs[k].spines['right'].set_visible(False)
    axs[k].spines['bottom'].set_visible(False)
    # axs[k].spines['left'].set_visible(False)
    axs[k].set_xticks([])
    axs[k].set_ylabel(None)
    # break

axs[0].set_ylabel("$\log p(\mathbf{x})$")


axs[1].text(0.1, 13.09, f"RQ-NSF\n(C) &\nRQ-NSF\n(AR)", color="black", va="center")
axs[1].scatter(0, 13.09, color="black")
axs[1].scatter(-0.2, 13.09, color="black")



axs[2].text(0.1, -13.9, f"RQ-NSF\n(AR)", color="black", va="bottom")
axs[2].scatter(0,-14.01, color="black")


axs[4].text(0.1, 0.66, f"Q-NSF\n(AR) &\nRQ-NSF\n(AR)", color="black", va="center")
axs[4].scatter(0,0.66, color="black")
axs[4].scatter(-0.2,0.66, color="black")






plt.tight_layout()
plt.savefig("resultsND_boxplot.pdf")
# plt.close()


# %%
df_table = df_all.drop_duplicates(["model_type", "dataset"]).pivot(index="dataset", columns="model_type", values="test_logprob_mean")
df_table = df_table[['DIFW (AR)', 'DIFW (CL)', 'BLOCK-NAF', 'FFJORD', 'GLOW', 'MAF', 'NAF',
        'Q-NSF (AR) ', 'Q-NSF (C)', 'RQ-NSF (AR) ', 'RQ-NSF (C) ', 'SOS']].drop(["SCURVE", "SWISSROLL"])

# %%
cols = df_table.columns
ncols = len(cols)
diff = np.zeros((ncols, ncols))
perc = np.zeros((ncols, ncols))
for i in range(ncols):
    for j in range(ncols):
        diff[i,j] = np.mean( (df_table[cols[i]] - df_table[cols[j]]) / df_table[cols[i]])
        perc[i,j] = np.sum( df_table[cols[i]] > df_table[cols[j]] ) / len(df_table)

# diff[np.triu_indices_from(diff, k=1)] = np.nan
diff = pd.DataFrame(diff, columns=cols, index=cols)

# perc[np.triu_indices_from(perc, k=1)] = np.nan
perc = pd.DataFrame(perc, columns=cols, index=cols)

# %%
from matplotlib.ticker import PercentFormatter

# sns.set_context("notebook")

f, ax = plt.subplots(figsize=(14/1.5, 10/1.5), constrained_layout=True)
sns.heatmap(diff, annot=True, linewidths=.5, ax=ax, fmt=".1%", 
    square=True, cmap="mako_r", annot_kws={"size": 9})
plt.yticks(rotation = 0)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
plt.xlabel(None)
plt.ylabel(None)
plt.savefig("resultsND_dataset_diff.pdf", bbox_inches="tight")
plt.close()

# %%

f, ax = plt.subplots(figsize=(14/1.5, 10/1.5), constrained_layout=True)
sns.heatmap(perc, annot=True, linewidths=.5, ax=ax, fmt=".0%", 
    square=True, cmap="mako_r", annot_kws={"size": 9})
plt.yticks(rotation = 0)
cbar = ax.collections[0].colorbar
cbar.ax.yaxis.set_major_formatter(PercentFormatter(1, 0))
plt.xlabel(None)
plt.ylabel(None)
plt.savefig("resultsND_dataset_perc.pdf", bbox_inches="tight")
plt.close()


# %% RANKING EVALUATION

# columns = ["Euclidean", "DBA", "SoftDTW", "DTAN", "ResNet-TW", "Ours"]
# m = df[columns].values

# # average accuracy
# accuracy_average = np.mean(m, axis=0)
# accuracy_median = np.median(m, axis=0)
# accuracy_std = np.std(m, axis=0)

# # rankings
# from scipy.stats import rankdata
# # order = 6 - np.argsort(m, axis=1)
# order = rankdata(-m, axis=1, method="min") # method = "ordinal" for previous results

# ranking_arithmetic = np.mean(order, axis=0) # average arithmetic ranking
# ranking_geometric = scipy.stats.gmean(order, axis=0) # average geometric ranking
# winning_times = np.sum(order == 1, axis=0)

# index_names = ["winning times", "ranking arithmetic", "ranking_geometric", "accuracy average"]
# pd.DataFrame([winning_times, ranking_arithmetic, ranking_geometric, accuracy_average], 
#     index=index_names, columns=columns).to_latex("ucr_dataset_ranking.tex", index=True, float_format='%10.6f')


