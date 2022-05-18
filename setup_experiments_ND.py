# %% CONFIGURATION ND

folder = ["resultsND"]
dataset = ["SCURVE", "SWISSROLL", "POWER", "MNIST", "MINIBOONE", "HEPMASS", "GAS", "BSDS300"]
dataset = ["POWER"]

train_size = [30000]
test_size = [5000]
batch_size = [512]
hidden_dim = [16]
hidden_layers = [4]
tess_size = [4,8,16]
flow_steps = [2,4,10]
epochs = [128]
lr = [5e-4]
model_type = ["CL", "AR"]

# TEST EXAMPLE
# hidden_dim = [1]
# hidden_layers = [0]
# tess_size = [4]
# flow_steps = [1]
# epochs = [2]

from itertools import product

hyperparameters = product(folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr, model_type)
k = 0
f = open("runND.sh", "w")
for params in hyperparameters:
    folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr, model_type = params
    command = """
python nf_ND_args.py  \
--folder {} \
--dataset {} \
--train-size {} \
--test-size {}  \
--batch-size {} \
--hidden-dim {} \
--hidden-layers {} \
--tess-size {} \
--flow-steps {} \
--epochs {} \
--lr {} \
--model-type {} """.format(folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr, model_type)
    f.write(command)
    f.write("\n")
    k += 1

f.close()

print("ND: prepared {} experiments".format(k))

