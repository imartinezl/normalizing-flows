# %%

import numpy as np
import gzip
import pickle

root = 'data/'

def one_hot_encode(labels, n_labels):
    # Transforms numeric labels to 1-hot encoded labels. Assumes numeric labels are in the range 0, 1, ..., n_labels-1.
    assert np.min(labels) >= 0 and np.max(labels) < n_labels

    y = np.zeros([labels.size, n_labels])
    y[range(labels.size), labels] = 1

    return y

def logit(x):
    # Elementwise logit (inverse logistic sigmoid)
    return np.log(x / (1.0 - x))

class MNIST:
    alpha = 1.0e-6

    class Data:
        def __init__(self, data, logit, dequantize, rng):

            x = self._dequantize(data[0], rng) if dequantize else data[0]  # dequantize pixels
            self.x = self._logit_transform(x) if logit else x              # logit
            self.labels = data[1]                                          # numeric labels
            self.y = one_hot_encode(self.labels, 10)                  # 1-hot encoded labels
            self.N = self.x.shape[0]                                       # number of datapoints

        @staticmethod
        def _dequantize(x, rng):
            """
            Adds noise to pixels to dequantize them.
            """
            return x + rng.rand(*x.shape) / 256.0

        @staticmethod
        def _logit_transform(x):
            """
            Transforms pixel values with logit to be unconstrained.
            """
            return logit(MNIST.alpha + (1 - 2*MNIST.alpha) * x)

    def __init__(self, logit=True, dequantize=True):

        # load dataset
        f = gzip.open(root + 'mnist/mnist.pkl.gz', 'rb')
        trn, val, tst = pickle.load(f, encoding='latin1')
        f.close()

        rng = np.random.RandomState(42)
        self.trn = self.Data(trn, logit, dequantize, rng)
        self.val = self.Data(val, logit, dequantize, rng)
        self.tst = self.Data(tst, logit, dequantize, rng)

        im_dim = int(np.sqrt(self.trn.x.shape[1]))
        self.n_dims = (1, im_dim, im_dim)
        self.n_labels = self.trn.y.shape[1]
        self.image_size = [im_dim, im_dim]


