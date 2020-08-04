from itertools import chain
from time import perf_counter as counter

import torch

from torch.nn import RNN, Linear
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD

from deepnp.models import *


class RNNTrainer:
    def __init__(self, input_dim, hidden_dim, output_size, backend='np'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.backend = backend

        D, H, V = input_dim, hidden_dim, output_size
        self.rnn_Wx = np.random.randn(D, H) / np.sqrt(D)
        self.rnn_Wh = np.random.randn(H, H) / np.sqrt(H)
        self.rnn_b = np.random.randn(H) / np.sqrt(H)
        self.fc_W = np.random.randn(H, V) / np.sqrt(H)
        self.fc_b = np.random.randn(V) / np.sqrt(V)

    def check_gpu(self):
        if GPU:
            if np.__name__ != 'cupy':
                import cupy as np
            np.cuda.Device(0).use()
            available = "using gpu:", np.cuda.is_available()
        else:
            available = "GPU is set false"
        print(available)
        return available

    def fit(self, X, y_true, batch_size, lr, n_epochs, print_many=False, verbose=1):
        """
        X = np.array([
            [[1, 2], [4, 6], [7, 9], [5, 1]],
            [[2, 1], [7, 9], [8, 9], [1, 2]],
            [[9, 8], [6, 4], [2, 1], [8, 4]],
            [[4, 8], [2, 7], [5, 7], [6, 3]],
        ])
        y_true = np.array([1, 2, 1, 3])
        """

        self.batch_size_global = batch_size
        self.total_size = len(X)
        self.time_steps = X.shape[1]
        self.max_iters = self.total_size // batch_size
        if self.total_size % self.batch_size_global != 0:
            self.max_iters += 1

        if self.backend in ['np', 'numpy', 'cupy', 'cp']:
            return self.train_np(X, y_true, batch_size, lr, n_epochs, print_many, verbose)
        elif self.backend in ['torch', 'pytorch']:
            return self.train_torch(X, y_true, batch_size, lr, n_epochs, print_many, verbose)

    def predict(self, x):
        if x.shape[1] != self.input_dim or x.ndim != 2:
            raise Exception("Dimension missmatch")

        x_batch = x.reshape(-1, 1, self.input_dim)
        batch_size_local = 1

        rnn = RNNLayer(batch_size=batch_size_local, input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                       Wx=self.rnn_Wx, Wh=self.rnn_Wh, bias=self.rnn_b)
        fc = FullyConnectedLayer(W=self.fc_W, bias=self.fc_b, batch_size=batch_size_local)
        h_last = rnn.forward(x_batch)
        fc_out = fc.forward(x=h_last)

        probs = softmax(fc_out)
        return np.argmax(probs, axis=1).item()

    def train_torch(self, X, y_true, batch_size, learning_rate, num_epochs, print_many, verbose):
        self.batch_size_global = batch_size
        progresses = {int(num_epochs // (100 / i)): i for i in range(1, 101, 1)}
        t0 = counter()
        durations = []

        device = torch.device('cuda:0')
        rnn = RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, nonlinearity='tanh',
                  bias=True, batch_first=False).to(device)
        fc = Linear(self.hidden_dim, self.output_size, bias=True).to(device)
        params = [rnn.parameters(), fc.parameters()]
        optimizer = SGD(chain(*params), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(self.max_iters):
                x_batch = X[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                x_batch = np.array([x_batch[:, step, :] for step in range(self.time_steps)])
                y_true_batch = y_true[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                batch_size_local = x_batch.shape[1]

                # convert to pytorch tensor
                y_true_batch = y_true_batch.astype(np.int64)
                y_true_batch = torch.tensor(y_true_batch, requires_grad=False).to(device)
                x_batch = x_batch.astype(np.float32)
                x_batch = torch.tensor(x_batch, requires_grad=True).to(device)

                # forward pass
                h_stack, h_last = rnn.forward(x_batch)
                fc_out = fc.forward(h_last)
                log_y_pred = F.log_softmax(input=fc_out, dim=2)
                log_y_pred = log_y_pred.view(batch_size_local, self.output_size)
                loss = F.nll_loss(input=log_y_pred, target=y_true_batch, reduction='mean')

                # update gradient
                optimizer.zero_grad()
                loss.backward()
                epoch_loss += loss.item()
                optimizer.step()

            durations.append(counter() - t0)
            t0 = counter()
            if (print_many and epoch % 100 == 0) or (not print_many and epoch in progresses):
                print(f"after epoch: {epoch}, epoch_losses: {round(epoch_loss / self.max_iters, 3)}")

        if verbose > 0:
            avg_epoch_time = sum(durations) / len(durations)
            print("average epoch time:", round(avg_epoch_time, 3))
            return avg_epoch_time

    def train_np(self, X, y_true, batch_size, learning_rate, num_epochs, print_many, verbose):
        self.batch_size_global = batch_size
        progresses = {int(num_epochs // (100 / i)): i for i in range(1, 101, 1)}
        t0 = counter()
        durations = []
        for epoch in range(num_epochs):
            epoch_losses = []
            for i in range(self.max_iters):
                x_batch = X[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                x_batch = np.array([x_batch[:, step, :] for step in range(self.time_steps)])
                y_true_batch = y_true[i * self.batch_size_global:(i + 1) * self.batch_size_global]
                batch_size_local = x_batch.shape[1]

                # self.rnn_h_init = np.zeros(shape=(batch_size_local, self.hidden_dim))
                rnn = RNNLayer(batch_size=batch_size_local, input_dim=self.input_dim, hidden_dim=self.hidden_dim,
                               Wx=self.rnn_Wx, Wh=self.rnn_Wh, bias=self.rnn_b)
                fc = FullyConnectedLayer(W=self.fc_W, bias=self.fc_b, batch_size=batch_size_local)
                loss = SoftmaxWithLossLayer()

                h_last = rnn.forward(x_batch)
                fc_out = fc.forward(x=h_last)
                loss_value = loss.forward(x=fc_out, y_true=y_true_batch)
                epoch_losses.append(loss_value)

                # backward pass
                d_L = loss.backward()
                fc_grads = fc.backward(d_L)
                d_fc_W = fc_grads['W_grad']
                d_fc_bias = fc_grads['bias_grad']
                d_h_last = fc_grads['x_grad']
                grads = rnn.backward(d_last_hidden=d_h_last)

                # parameter update
                lr = learning_rate
                self.rnn_Wx -= lr * grads["Wx_grad"]
                self.rnn_Wh -= lr * grads["Wh_grad"]
                self.rnn_b -= lr * grads["bias_grad"]
                # self.rnn_h_init -= lr * grads['h_init_grad']
                self.fc_W -= lr * d_fc_W
                self.fc_b -= lr * d_fc_bias

            durations.append(counter() - t0)
            t0 = counter()
            if (print_many and epoch % 100 == 0) or (not print_many and epoch in progresses):
                print(f"after epoch: {epoch}, epoch_losses: {round(np.mean(np.array(epoch_losses)).item(), 3)}")

        if verbose > 0:
            avg_epoch_time = sum(durations) / len(durations)
            print("average epoch time:", round(avg_epoch_time, 3))
            return avg_epoch_time
