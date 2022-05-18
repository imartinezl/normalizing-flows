# %% CONFIGURATION 1D

folder = ["results1D"]
dataset = ["UNIFORM", "GAUSSIANMIXTURE", "GAUSSIAN", "POWER", "BLOBS"]
train_size = [5000]
test_size = [2000]
batch_size = [256]
tess_size = [4,8,16,32]
flow_steps = [1,2,3]
epochs = [500]
lr = [1e-4, 1e-3, 5e-3]

from itertools import product

hyperparameters = product(folder, dataset, train_size, test_size, batch_size, tess_size, flow_steps, epochs, lr)
k = 0
f = open("run1D.sh", "w")
for params in hyperparameters:
    folder, dataset, train_size, test_size, batch_size, tess_size, flow_steps, epochs, lr = params
    command = """
python nf_1D_args.py  \
--folder {} \
--dataset {} \
--train-size {} \
--test-size {}  \
--batch-size {} \
--tess-size {} \
--flow-steps {} \
--epochs {} \
--lr {} """.format(folder, dataset, train_size, test_size, batch_size, tess_size, flow_steps, epochs, lr)
    f.write(command)
    f.write("\n")
    k += 1

f.close()

print("1D: prepared {} experiments".format(k))