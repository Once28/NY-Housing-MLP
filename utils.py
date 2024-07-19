import numpy as np
from data_preprocessing import  load_and_preprocess_data
# import pdb

# Softmax loss and Softmax gradient
### Loss functions ###

def data_loader():
    Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, num_classes = load_and_preprocess_data()

    idx_order = np.random.permutation(Xtrain.shape[0])
    Xtrain = Xtrain[idx_order]
    Ytrain = Ytrain[idx_order]

    num_classes = len(np.unique(np.concatenate([Ytrain, Yval, Ytest])))

    return Xtrain, Ytrain, Xval, Yval, Xtest, Ytest, num_classes

class softmax_cross_entropy:
    def __init__(self, num_classes=None):
        self.expand_Y = None
        self.calib_logit = None
        self.sum_exp_calib_logit = None
        self.prob = None
        self.num_classes = num_classes

    def forward(self, X, Y):
        self.num_classes = X.shape[1] if self.num_classes is None else self.num_classes
        Y = Y.astype(int)
        self.expand_Y = np.zeros((X.shape[0], self.num_classes))
        rows = np.arange(X.shape[0])
        self.expand_Y[rows, Y] = 1.0
        self.calib_logit = X - np.max(X, axis=1, keepdims=True)
        self.calib_logit = self.calib_logit.astype(float)
        self.sum_exp_calib_logit = np.sum(np.exp(self.calib_logit), axis=1, keepdims=True)
        self.prob = np.exp(self.calib_logit) / self.sum_exp_calib_logit
        forward_output = -np.sum(self.expand_Y * np.log(self.prob + 1e-9)) / X.shape[0]
        return forward_output

    def backward(self, X, Y):
        Y = Y.astype(int)
        backward_output = np.zeros_like(X)
        backward_output = self.prob - self.expand_Y
        backward_output /= X.shape[0]
        return backward_output

def predict_label(f):
    if f.shape[1] == 1:
        return (f > 0).astype(float)
    else:
        return np.argmax(f, axis=1).astype(float)
class DataSplit:
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y
        self.N, self.d = self.X.shape

    def get_example(self, idx):
        batchX = self.X[idx]
        batchY = self.Y[idx].reshape(-1, 1)
        return batchX, batchY
