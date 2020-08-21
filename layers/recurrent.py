from collections import namedtuple, OrderedDict

from deepnp.functions import *
import deepnp.initializers as init


class RNNCell:
    def __init__(self, batch_size, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        N, D, H = batch_size, input_dim, hidden_dim
        Wx = init.normal(D, H) if Wx is None else Wx
        Wh = init.normal(H, H) if Wh is None else Wh
        bias = init.normal(H) if bias is None else bias
        self.batch_size = N
        self.parameters = [Wx, Wh, bias]
        self.grads = {'Wx': np.zeros_like(Wx),
                      'Wh': np.zeros_like(Wh),
                      'bias': np.zeros_like(bias),
                      'h_prev': np.zeros(shape=(N, H))}
        self.cache = {'x_t': None,
                      'h_prev': None,
                      'h_next': None}

    def forward(self, x_t, h_prev):
        Wx, Wh, bias = self.parameters
        linear = np.matmul(x_t, Wx) + np.matmul(h_prev, Wh) + bias
        h_next = np.tanh(linear)
        self.cache = {'x_t': x_t,
                      'h_prev': h_prev,
                      'h_next': h_next}
        return h_next

    def backward(self, d_h_next=1, optimize=True):
        Wx, Wh, bias = self.parameters
        x_t, h_prev, h_next = self.cache['x_t'], self.cache['h_prev'], self.cache['h_next']

        d_linear = d_h_next * (1 - np.square(h_next))  # element-wise dot product
        d_bias = np.sum(d_linear, axis=0, keepdims=False)
        d_Wx = np.matmul(x_t.T, d_linear)
        d_Wh = np.matmul(h_prev.T, d_linear)
        d_h_prev = np.matmul(d_linear, Wh.T)

        if not optimize:
            d_x_t = np.matmul(d_linear, Wx.T)

        self.grads['Wx'][...] = d_Wx
        self.grads['Wh'][...] = d_Wh
        self.grads['bias'][...] = d_bias
        self.grads['h_prev'][...] = d_h_prev

        return self.grads


class RNNLayer:
    def __init__(self, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        D, H = input_dim, hidden_dim
        self.input_dim, self.hidden_dim = D, H
        Wx = init.normal(D, H) if Wx is None else Wx
        Wh = init.normal(H, H) if Wh is None else Wh
        bias = init.normal(H) if bias is None else bias
        self.parameters = [Wx, Wh, bias]
        self.grads = {'Wx': np.zeros_like(Wx),
                      'Wh': np.zeros_like(Wh),
                      'bias': np.zeros_like(bias)}
        self.timestep_cells = []

    def update(self, parameters):
        self.parameters = parameters
        self.timestep_cells = list()

    def forward(self, x_sequence, h_init=None):
        batch_size, timesteps, input_dim = x_sequence.shape
        N, T, D, H = batch_size, timesteps, input_dim, self.hidden_dim
        Wx, Wh, bias = self.parameters
        h_prev = init.zeros(N, H) if h_init is None else h_init

        # N*T*D Style
        h_stack = init.empty(N, T, H)
        for t in range(T):
            timestep_cell = RNNCell(N, D, H, Wx=Wx, Wh=Wh, bias=bias)
            h_next = timestep_cell.forward(x_sequence[:, t, :], h_prev)  # 해당 timestep마다의 (N, D) 형태 2차원 텐서, h_prev
            self.timestep_cells.append(timestep_cell)
            h_stack[:, t, :] = h_next  # h_next는 (N, H)의 한 timestep rnn 결과물
            h_prev = h_next

        # T*N*D Style (Deprecated)
        # for time_step, x_t in enumerate(x_sequence):
        #     timestep_cell = RNNCell(self.N, self.D, self.hidden_dim, Wx=Wx, Wh=Wh, bias=bias)
        #     h_next = timestep_cell.forward(x_t, h_prev)
        #     self.rnn_steps.append(timestep_cell)
        #     h_prev = h_next

        h_last = h_next
        return h_last, h_stack

    def backward(self, d_h_next=1, optimize=True):
        Wx, Wh, bias = self.parameters
        d_Wx = np.zeros_like(Wx)
        d_Wh = np.zeros_like(Wh)
        d_bias = np.zeros_like(bias)

        for idx, layer in enumerate(reversed(self.timestep_cells)):
            grad = layer.backward(d_h_next=d_h_next, optimize=optimize)
            d_Wx += grad['Wx']
            d_Wh += grad['Wh']
            d_bias += grad['bias']
            d_h_next = grad['h_prev']

        for d in [d_Wx, d_Wh, d_bias]:
            np.clip(d, -1, 1, out=d)

        self.grads['Wx'][...] = d_Wx
        self.grads['Wh'][...] = d_Wh
        self.grads['bias'][...] = d_bias

        return self.grads


class RNNLayerWithTimesteps:
    def __init__(self, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        D, H = input_dim, hidden_dim
        self.input_dim, self.hidden_dim = D, H
        Wx = init.normal(D, H) if Wx is None else Wx
        Wh = init.normal(H, H) if Wh is None else Wh
        bias = init.normal(H) if bias is None else bias
        self.parameters = [Wx, Wh, bias]
        self.grads = {'Wx': np.zeros_like(Wx),
                      'Wh': np.zeros_like(Wh),
                      'bias': np.zeros_like(bias)}
        self.timestep_cells = []

    def update(self, *parameters):
        self.parameters = parameters
        self.timestep_cells = list()

    def forward(self, x_sequence, h_init=None):
        batch_size, timesteps, input_dim = x_sequence.shape
        N, T, D, H = batch_size, timesteps, input_dim, self.hidden_dim
        Wx, Wh, bias = self.parameters
        h_prev = init.zeros(N, H) if h_init is None else h_init

        # N*T*D Style
        h_stack = init.empty(N, T, H)
        for t in range(T):
            timestep_cell = RNNCell(N, D, H, Wx=Wx, Wh=Wh, bias=bias)
            h_next = timestep_cell.forward(x_sequence[:, t, :], h_prev)  # 해당 timestep마다의 (N, D) 형태 2차원 텐서, h_prev
            self.timestep_cells.append(timestep_cell)
            h_stack[:, t, :] = h_next  # h_next는 (N, H)의 한 timestep rnn 결과물
            h_prev = h_next

        h_last = h_next
        return h_last, h_stack

    def backward(self, d_h_stack, optimize=True):
        Wx, Wh, bias = self.parameters
        N, T, H = d_h_stack.shape
        D, H = Wx.shape

        d_Wx = np.zeros_like(Wx)
        d_Wh = np.zeros_like(Wh)
        d_bias = np.zeros_like(bias)
        d_h_prev = init.zeros(N, H)

        for t in reversed(range(T)):
            if t == 0: continue
            # d_h_next = 0.2 * d_h_stack[:, t, :] + d_h_prev
            cell = self.timestep_cells[t]
            d_h_next = d_h_stack[:, t, :] + d_h_prev
            grad = cell.backward(d_h_next=d_h_next, optimize=optimize)
            d_Wx += grad['Wx']
            d_Wh += grad['Wh']
            d_bias += grad['bias']
            d_h_prev[...] = grad['h_prev']

        for d in [d_Wx, d_Wh, d_bias]:
            np.clip(d, -1, 1, out=d)

        self.grads['Wx'][...] = d_Wx
        self.grads['Wh'][...] = d_Wh
        self.grads['bias'][...] = d_bias

        return self.grads


class LSTMCell:
    def __init__(self, batch_size, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        N, D, H = batch_size, input_dim, hidden_dim
        Wx = init.simplexavier(D, 4 * H) if Wx is None else Wx
        Wh = init.simplexavier(H, 4 * H) if Wh is None else Wh
        bias = init.simplexavier(4 * H) if bias is None else bias
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.parameters = [Wx, Wh, bias]
        self.grads = {'Wx': np.zeros_like(Wx),
                      'Wh': np.zeros_like(Wh),
                      'bias': np.zeros_like(bias)}
        self.cache = {}

    def forward(self, x, c_prev, h_prev):
        # x_t : (N, D) c_prev, h_prev : (N, H)
        # f : forget gate, n : new info, i : input gate, o : output gate
        H = self.hidden_dim
        Wx, Wh, bias = self.parameters
        fc = x @ Wx + h_prev @ Wh + bias  # N * 4H

        f = sigmoid(fc[:, :H])
        n = np.tanh(fc[:, H:2 * H])
        i = sigmoid(fc[:, 2 * H:3 * H])
        o = sigmoid(fc[:, 3 * H:])

        c_next = (c_prev * f) + (n * i)
        h_next = np.tanh(c_next) * o
        self.cache = (x, c_prev, h_prev, c_next, h_next, f, n, i, o,)
        return c_next, h_next

    def backward(self, d_c_next, d_h_next):
        # d는 모두 N, H. 원하는 건 dWx, dWh, dbias
        # d_h_prev, d_c_prev 구해서 보내줘야함
        # f, n, i, o 순으로 구한 뒤 np.hstack()으로 쌓기
        Wx, Wh, bias = self.parameters
        x, c_prev, h_prev, c_next, h_next, f, n, i, o = self.cache

        tanh_c_next = np.tanh(c_next)
        d_c_next_agg = d_c_next + (d_h_next * o) * (1 - tanh_c_next ** 2)

        d_f = d_c_next_agg * c_prev
        d_n = d_c_next_agg * i
        d_i = d_c_next_agg * n
        d_o = d_h_next * tanh_c_next

        d_f *= f * (1 - f)
        d_n *= 1 - n ** 2
        d_i *= i * (1 - i)
        d_o *= o * (1 - o)

        d_fc = np.hstack((d_f, d_n, d_i, d_o))
        d_Wx = x.T @ d_fc
        d_Wh = h_prev.T @ d_fc
        d_bias = d_fc.sum(axis=0)

        self.grads['Wx'][...] = d_Wx
        self.grads['Wh'][...] = d_Wh
        self.grads['bias'][...] = d_bias

        d_c_prev = d_c_next_agg * f
        d_h_prev = d_fc @ Wh.T
        d_x = d_fc @ Wx.T

        return d_x, d_c_prev, d_h_prev


class LSTMLayerTimesteps:
    def __init__(self, input_dim, hidden_dim, Wx=None, Wh=None, bias=None, stateful=False):
        self.input_dim, self.hidden_dim = input_dim, hidden_dim
        D, H = input_dim, hidden_dim
        Wx = init.simplexavier(D, 4 * H) if Wx is None else Wx
        Wh = init.simplexavier(H, 4 * H) if Wh is None else Wh
        bias = init.simplexavier(4 * H) if bias is None else bias
        self.cell_state, self.hidden_state, self.d_h_0 = None, None, None
        self.stateful = stateful
        self.parameters = [Wx, Wh, bias]
        self.grads = [np.zeros_like(Wx),
                      np.zeros_like(Wh),
                      np.zeros_like(bias)]
        # self.grads = {'Wx': np.zeros_like(Wx),
        #               'Wh': np.zeros_like(Wh),
        #               'bias': np.zeros_like(bias)}
        self.timestep_cells = list()

    def update(self, *parameters):
        self.parameters = parameters
        self.timestep_cells = list()

    def set_state(self, h, c=None):
        self.cell_state, self.hidden_state = c, h

    def reset_state(self):
        self.cell_state, self.hidden_state = None, None

    def reset_timesteps(self):
        self.timestep_cells = list()

    def forward(self, x_seq):
        N, T, D = x_seq.shape
        H = self.hidden_dim
        Wx, Wh, bias = self.parameters

        c_prev = init.zeros(N, H) if not self.stateful or self.cell_state is None else self.cell_state
        h_prev = init.zeros(N, H) if not self.stateful or self.hidden_state is None else self.hidden_state

        # x_seq : N, T, D
        h_stack = init.empty(N, T, H)
        for t in range(T):
            timestep_cell = LSTMCell(N, D, H, Wx, Wh, bias)
            c_next, h_next = timestep_cell.forward(x_seq[:, t, :], c_prev, h_prev)
            self.timestep_cells.append(timestep_cell)
            h_stack[:, t, :] = h_next
            c_prev, h_prev = c_next, h_next

        self.cell_state, self.hidden_state = c_next, h_next
        h_last = h_next
        return h_last, h_stack

    def backward(self, d_h_stack, optimize=True):
        Wx, Wh, bias = self.parameters
        N, T, H = d_h_stack.shape
        D, _ = Wx.shape

        d_Wx = np.zeros_like(Wx)
        d_Wh = np.zeros_like(Wh)
        d_bias = np.zeros_like(bias)
        d_h_prev = init.zeros(N, H)
        d_c_next = init.zeros(N, H)

        d_x_stack = init.empty(N, T, D)
        for t in reversed(range(T)):
            cell = self.timestep_cells[t]
            # d_h_next = 0.2 * d_h_stack[:, t, :] + d_h_prev
            d_h_next = d_h_stack[:, t, :] + d_h_prev
            d_x, d_c_prev, d_h_prev = cell.backward(d_c_next, d_h_next)
            d_x_stack[:, t, :] = d_x
            d_Wx += cell.grads['Wx']
            d_Wh += cell.grads['Wh']
            d_bias += cell.grads['bias']

        for d in [d_Wx, d_Wh, d_bias]:
            np.clip(d, -1, 1, out=d)

        self.grads[0][...] = d_Wx
        self.grads[1][...] = d_Wh
        self.grads[2][...] = d_bias

        self.d_h_0 = d_h_prev
        return d_x_stack


class FCLayerTimesteps:
    def __init__(self, W, bias):
        self.params = [W, bias]
        self.grads = [np.zeros_like(W),
                      np.zeros_like(bias)]
        # self.grads = {'W': np.zeros_like(W),
        #               'bias': np.zeros_like(bias),
        #               'x': None}
        self.x_flat = None
        self.input_dims = None

    def forward(self, x):
        W, bias = self.params
        N, T, H = x.shape
        assert H == W.shape[0]
        self.input_dims = (N, T, H,)

        x_flat = x.reshape(N * T, H)
        out_flat = np.matmul(x_flat, W) + bias
        out = out_flat.reshape(N, T, -1)

        self.x_flat = x_flat
        return out

    def backward(self, dout):
        W, bias = self.params
        N, T, H = self.input_dims
        dout = dout.reshape(N * T, -1)

        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x_flat.T, dout)
        db = np.sum(dout, axis=0)

        assert dx.size / (N * T) == H
        dx = dx.reshape(N, T, H)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        # self.grads['x'] = dx

        return dx


class SoftmaxWithLossLayerTimesteps:
    def __init__(self):
        self.y_pred = None
        self.y_true_flat = None
        self.input_dims = None

    def forward(self, x, y_true):
        """
        :param x: (N, T, V) 3-dimensional
        :param y_true: (N, T) 2-dimensional
        :return:
        """
        N, T, V = x.shape
        assert N == y_true.shape[0] and T == y_true.shape[1]
        assert x.ndim == 3 and y_true.ndim == 2
        self.input_dims = (N, T, V,)
        x_flat = x.reshape(N * T, V)
        y_true_flat = y_true.reshape(1, N * T)
        y_pred = softmax(x_flat)
        average_loss, num_acc = cross_entropy_error(y_pred, y_true_flat)
        self.y_pred, self.y_true_flat = y_pred, y_true_flat
        return average_loss, num_acc

    def backward(self, d_out=1):
        N, T, V = self.input_dims
        dx = self.y_pred.copy()  # N*T, V
        dx[np.arange(N * T), self.y_true_flat] -= 1  # N*T
        dx *= d_out
        dx /= N * T  # batch size로 나눔
        dx = dx.reshape(N, T, V)
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads
        dW[...] = 0
        # np.add.at(dW, self.idx, dout)
        dW[self.idx, :] += dout
        return None


class EmbeddingTimesteps:
    def __init__(self, W):
        self.params = [W]
        self.grad = [np.zeros_like(W)]
        self.layers = None
        self.W = W

    def forward(self, xs):
        N, T = xs.shape
        V, D = self.W.shape

        out = np.empty((N, T, D), dtype='f')
        self.layers = []

        for t in range(T):
            layer = Embedding(self.W)
            out[:, t, :] = layer.forward(xs[:, t])
            self.layers.append(layer)

        return out

    def backward(self, dout):
        N, T, D = dout.shape

        grad = 0
        for t in range(T):
            layer = self.layers[t]
            layer.backward(dout[:, t, :])
            grad += layer.grads[0]

        self.grad[0][...] = grad
        return None
