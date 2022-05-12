# %% CONFIGURATION 2D

folder = ["results2D"]
dataset = ["MOONS", "CIRCLES", "CRESCENT", "CRESCENTCUBED", "SINEWAVE", "ABS", "SIGN", "TWOSPIRALS", "CHECKERBOARD", "FOURCIRCLES", "DIAMOND"]
train_size = [5000]
test_size = [2000]
batch_size = [256]
hidden_dim = [8,16]
hidden_layers = [1,2,4]
tess_size = [4,8,16,32]
flow_steps = [1,2,3]
epochs = [500]
lr = [1e-3]

# TEST EXAMPLE
# hidden_dim = [1]
# hidden_layers = [0]
# tess_size = [4]
# flow_steps = [1]
# epochs = [2]

from itertools import product

hyperparameters = product(folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr)
k = 0
f = open("run2D.sh", "w")
for params in hyperparameters:
    folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr = params
    command = """
python nf_2D_args.py  \
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
--lr {} """.format(folder, dataset, train_size, test_size, batch_size, hidden_dim, hidden_layers, tess_size, flow_steps, epochs, lr)
    f.write(command)
    f.write("\n")
    k += 1

f.close()

print("2D: prepared {} experiments".format(k))

