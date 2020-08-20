from itertools import chain
from time import perf_counter as counter

import torch

from torch.nn import RNN, Linear
import torch.nn.functional as F
from torch.autograd import Variable
from torch.optim import SGD, Adam

from deepnp.layers import *


class RNNTrainer:
    def __init__(self, input_dim, hidden_dim, output_size, backend='np', timemethod='stack'):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_size = output_size
        self.backend = backend
        self.timemethod = timemethod

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

    def fit(self, X, y_true, n_epochs, lr, batch_size, print_many=False, verbose=1):
        """
        X = N, T, D
        y_true = np.array([1, 2, 1, 3])
        """

        self.total_size = len(X)
        self.batch_size = batch_size
        self.time_steps = X.shape[1]
        self.max_iters = self.total_size // batch_size
        if self.total_size % self.batch_size != 0:
            self.max_iters += 1

        if self.backend in ['np', 'numpy', 'cupy', 'cp']:
            X, y_true = np.asarray(X), np.asarray(y_true)
            if self.timemethod in ['shallow', 'last']:
                return self.train_np_last(X, y_true, batch_size, lr, n_epochs, print_many, verbose)
            elif self.timemethod in ['stack', 'all', 'timesteps', 'timestep']:
                return self.train_np_stack(X, y_true, batch_size, lr, n_epochs, print_many, verbose)
            else:
                raise
        elif self.backend in ['torch', 'pytorch']:
            return self.train_torch(X, y_true, batch_size, lr, n_epochs, print_many, verbose)

    def predict(self, x):
        # x는 T, D
        if x.shape[1] != self.input_dim or x.ndim != 2:
            raise Exception("Dimension missmatch")

        x = np.asarray(x)
        x_batch = x.reshape(1, -1, self.input_dim)
        batch_size_local = 1

        rnn = RNNLayer(self.input_dim, self.hidden_dim, Wx=self.rnn_Wx, Wh=self.rnn_Wh, bias=self.rnn_b)
        fc = FCLayer(W=self.fc_W, bias=self.fc_b, batch_size=1)
        h_last, h_stack = rnn.forward(x_batch)
        fc_out = fc.forward(x=h_last)

        probs = softmax(fc_out)
        return np.argmax(probs, axis=1).item()

    def train_torch(self, X, y_true, batch_size, learning_rate, num_epochs, print_many, verbose):
        self.batch_size = batch_size
        progresses = {int(num_epochs // (100 / i)): i for i in range(1, 101, 1)}
        t0 = counter()
        durations = []

        device = torch.device('cuda:0')
        rnn = RNN(input_size=self.input_dim, hidden_size=self.hidden_dim, num_layers=1, nonlinearity='tanh',
                  bias=True, batch_first=False).to(device)
        fc = FCLayer(self.hidden_dim, self.output_size, bias=True).to(device)
        params = [rnn.parameters(), fc.parameters()]
        optimizer = SGD(chain(*params), lr=learning_rate)
        for epoch in range(num_epochs):
            epoch_loss = 0
            for i in range(self.max_iters):
                x_batch = X[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch = np.array([x_batch[:, step, :] for step in range(self.time_steps)])
                y_true_batch = y_true[i * self.batch_size:(i + 1) * self.batch_size]
                batch_size_local = x_batch.shape[1]

                # convert to pytorch tensor
                y_true_batch = y_true_batch.astype(np.int64)
                y_true_batch = torch.tensor(y_true_batch, requires_grad=False).to(device)
                x_batch = x_batch.astype(np.float32)
                x_batch = torch.tensor(x_batch, requires_grad=True).to(device)

                # forward pass
                h_stack, h_last = rnn.forward(x_batch, hx=None)
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

    def train_np_stack(self, X, y_true, batch_size, learning_rate, num_epochs, print_many, verbose):
        self.batch_size = batch_size
        lr = learning_rate
        progresses = {int(num_epochs // (100 / i)): i for i in range(1, 101, 1)}
        t0 = counter()
        durations = []

        rnn = RNNLayerWithTimesteps(input_dim=self.input_dim, hidden_dim=self.hidden_dim, Wx=self.rnn_Wx,
                                    Wh=self.rnn_Wh, bias=self.rnn_b)

        for epoch in range(num_epochs):
            epoch_losses, epoch_acc = [], []
            for i in range(self.max_iters):
                # N*T*D batch style
                x_batch = X[i * self.batch_size: (i + 1) * self.batch_size]
                y_true_batch = y_true[i * self.batch_size:(i + 1) * self.batch_size]
                current_batch_size = x_batch.shape[0]

                fc = FCLayerTimesteps(W=self.fc_W, bias=self.fc_b)
                loss = SoftmaxWithLossLayerTimesteps()

                h_last, h_stack = rnn.forward(x_batch)
                fc_out = fc.forward(x=h_stack)
                loss_value, num_acc = loss.forward(x=fc_out, y_true=y_true_batch)
                epoch_losses.append(loss_value)
                epoch_acc.append(num_acc)

                # backward pass
                d_L = loss.backward()
                d_h_stack = fc.backward(d_L)
                rnn.backward(d_h_stack=d_h_stack, optimize=True)

                # parameter update
                self.rnn_Wx -= lr * rnn.grads["Wx_grad"]
                self.rnn_Wh -= lr * rnn.grads["Wh_grad"]
                self.rnn_b -= lr * rnn.grads["bias_grad"]
                self.fc_W -= lr * fc.grads['W_grad']
                self.fc_b -= lr * fc.grads['bias_grad']

                rnn.update(self.rnn_Wx, self.rnn_Wh, self.rnn_b)

            durations.append(counter() - t0)
            t0 = counter()
            if (print_many and epoch % 100 == 0) or (not print_many and epoch in progresses):
                acc_s = f"{round(sum(epoch_acc) / (X.shape[0] * X.shape[1]) * 100, 4)}%"
                loss_s = round(np.mean(np.array(epoch_losses)).item(), 3)
                perp = round(np.exp(loss_s).item(), 2)
                print(f"epoch: {epoch}, loss: {loss_s}, perp: {perp}, acc: {acc_s}")

        if verbose > 0:
            avg_epoch_time = sum(durations) / len(durations)
            print("average epoch time:", round(avg_epoch_time, 3))
            return avg_epoch_time

    def train_np_last(self, X, y_true, batch_size, learning_rate, num_epochs, print_many, verbose):
        self.batch_size = batch_size
        lr = learning_rate
        progresses = {int(num_epochs // (100 / i)): i for i in range(1, 101, 1)}
        t0 = counter()
        durations = []

        rnn = RNNLayer(input_dim=self.input_dim, hidden_dim=self.hidden_dim, Wx=self.rnn_Wx, Wh=self.rnn_Wh,
                       bias=self.rnn_b)

        for epoch in range(num_epochs):
            epoch_losses, epoch_acc = [], []
            for i in range(self.max_iters):
                # # T*N*D batch style
                # x_batch = X[i * self.batch_size: (i + 1) * self.batch_size]
                # x_batch = np.array([x_batch[:, step, :] for step in range(self.time_steps)])
                # y_true_batch = y_true[i * self.batch_size:(i + 1) * self.batch_size]
                # current_batch_size = x_batch.shape[1]

                # N*T*D batch style
                x_batch = X[i * self.batch_size: (i + 1) * self.batch_size]
                y_true_batch = y_true[i * self.batch_size:(i + 1) * self.batch_size]
                current_batch_size = x_batch.shape[0]

                fc = FCLayer(W=self.fc_W, bias=self.fc_b, batch_size=current_batch_size)
                loss = SoftmaxWithLossLayer()

                h_last, h_stack = rnn.forward(x_batch)
                fc_out = fc.forward(x=h_last)
                loss_value, num_acc = loss.forward(x=fc_out, y_true=y_true_batch)
                epoch_losses.append(loss_value)
                epoch_acc.append(num_acc)

                # backward pass
                d_L = loss.backward()
                fc_grads = fc.backward(d_L)
                d_fc_W = fc_grads['W_grad']
                d_fc_bias = fc_grads['bias_grad']
                d_h_last = fc_grads['x_grad']
                grads = rnn.backward(d_h_next=d_h_last, optimize=True)

                # parameter update
                self.rnn_Wx -= lr * grads["Wx_grad"]
                self.rnn_Wh -= lr * grads["Wh_grad"]
                self.rnn_b -= lr * grads["bias_grad"]
                self.fc_W -= lr * d_fc_W
                self.fc_b -= lr * d_fc_bias

                parameters = [self.rnn_Wx, self.rnn_Wh, self.rnn_b]
                rnn.update(parameters)

            durations.append(counter() - t0)
            t0 = counter()
            if (print_many and epoch % 100 == 0) or (not print_many and epoch in progresses):
                acc_s = f"{round(sum(epoch_acc) / (X.shape[0]) * 100, 4)}%"
                loss_s = round(np.mean(np.array(epoch_losses)).item(), 3)
                perp = round(np.exp(loss_s).item(), 2)
                print(f"epoch: {epoch}, loss: {loss_s}, perp: {perp}, acc: {acc_s}")

        if verbose > 0:
            avg_epoch_time = sum(durations) / len(durations)
            print("average epoch time:", round(avg_epoch_time, 3))
            return avg_epoch_time
