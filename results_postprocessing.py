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
