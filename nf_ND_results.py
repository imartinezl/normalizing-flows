# %%

import os
import json
import pandas as pd
import torch

datasets = ["SCURVE", "SWISSROLL", "POWER", "MNIST", "MINIBOONE", "HEPMASS", "GAS", "BSDS300"]

folder = "resultsND"
results = []
for dataset in datasets:
    runs = os.listdir(os.path.join(folder, dataset))

    for run in runs:
        path = os.path.join(folder, dataset, run)

        if "loss.pdf" not in os.listdir(path):
            print("Bad run" + str(run))
            continue
        
        with open(os.path.join(path, 'config.json'), 'r') as fp:
            config = json.load(fp)
        config["run"] = run

        checkpoints = [fp for fp in os.listdir(path) if "epoch" in fp]
        # for checkpoint in checkpoints:
        # checkpoint = torch.load(os.path.join(path, checkpoints[-1]))
        checkpoint = torch.load(os.path.join(path, "epoch_best.pth"))

        keys = [
            'train_loss', 'train_logprob_mean', 'train_logprob_std',
            'test_loss', 'test_logprob_mean' ,'test_logprob_std'
        ]

        results.append(dict(config, **{k: checkpoint[k] for k in keys}))

pd.DataFrame(results).to_csv("resultsND.csv", index=False)

