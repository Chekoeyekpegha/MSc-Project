import numpy as np
import math
import torch
from torch import nn
from river import neural_net as river_nn
from river import optim as river_optim


class ExStream(nn.Module):
    def __init__(self, shape, capacity, lr=2e-3, weight_decay=0.0, dropout=0.0, batch_norm=True,
                 activation='relu', seed=111):
        super(ExStream, self).__init__()

        self.input_shape = shape[0]
        self.num_classes = shape[-1]
        self.capacity = capacity
        self.buffers = 'class'  # Use class-specific buffers
        self.lr = lr
        self.weight_decay = weight_decay
        self.idx = 0

        # Initialize buffer arrays
        full_capacity = capacity * self.num_classes
        self.X_b = torch.zeros((full_capacity, self.input_shape))
        self.y_b = torch.zeros(full_capacity, dtype=torch.long)
        self.c_b = torch.zeros(full_capacity)
        self.buffer_counts = torch.zeros(self.num_classes)

        # Make the MLP classifier using River's neural_net
        self.classifier = self.make_mlp_classifier(shape, activation, batch_norm, dropout, seed)

        # Make the optimizer using River's optim module
        self.optimizer = river_optim.Adam(self.classifier.parameters(), lr=lr, weight_decay=weight_decay)

    def init_weights(self, m):
        if type(m) == river_nn.Linear:
            size = m.w.shape
            fan_out = size[0]
            fan_in = size[1]
            river_nn.initializers.Normal(mean=0., scale=math.sqrt(2 / (fan_in + fan_out))).initialize(m.w)
            m.b.set(1)

    def forward(self, x):
        return self.classifier.predict_one(x)

    def make_mlp_classifier(self, shape, activation, batch_norm, dropout, seed):
        river_nn.set_seed(seed)
        classifier = river_nn.Sequential()

        for i in range(len(shape) - 2):
            layer = river_nn.Linear(shape[i], shape[i + 1])
            layer.apply(self.init_weights)
            act = river_nn.ReLU()

            if batch_norm:
                if i != len(shape) - 1:
                    layer = river_nn.Sequential(layer, river_nn.BatchNorm(), act, river_nn.Dropout(dropout))
                else:
                    layer = river_nn.Sequential(layer, river_nn.BatchNorm(), act)
            else:
                if i != len(shape) - 1:
                    layer = river_nn.Sequential(layer, act, river_nn.Dropout(dropout))
                else:
                    layer = river_nn.Sequential(layer, act)

            classifier.append(layer)

        layer = river_nn.Linear(shape[-2], shape[-1])
        layer.apply(self.init_weights)
        classifier.append(layer)

        return classifier

    def fit(self, X, y):
        if self.idx > 1:
            X_con = self.X_b[0:self.idx, :]
            y_con = self.y_b[0:self.idx]
            self._consolidate_one_epoch(X_con, y_con)

    def predict(self, X):
        samples = X.shape[0]
        output = torch.zeros((samples, self.num_classes))
        for i in range(samples):
            x = X[i, :]
            preds = self.classifier.predict_one(x)
            output[i, preds] = 1
        preds = output.argmax(dim=1)
        return preds.numpy()

    def _consolidate_one_epoch(self, X, y):
        for i in range(X.shape[0]):
            x = X[i, :]
            label = y[i]
            self.classifier.learn_one(x, label)

    def _l2_dist_metric(self, H):
        M, d = H.shape
        H2 = torch.reshape(H, (M, 1, d))
        inside = H2 - H
        square_sub = torch.mul(inside, inside)
        psi = torch.sum(square_sub, dim=2)

        mb = psi.shape[0]
        diag_vec = torch.ones(mb) * np.inf
        mask = torch.diag(torch.ones_like(diag_vec))
        psi = mask * torch.diag(diag_vec) + (1. - mask) * psi

        idx = torch.argmin(psi)
        idx_row = idx / mb
        idx_col = idx % mb

        return torch.min(idx_row, idx_col), torch.max(idx_row, idx_col)
