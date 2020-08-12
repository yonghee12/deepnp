from collections import namedtuple

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
        self.grads = {'Wx_grad': np.zeros_like(Wx),
                      'Wh_grad': np.zeros_like(Wh),
                      'bias_grad': np.zeros_like(bias),
                      'h_prev_grad': np.zeros(shape=(N, H))}
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

        if not optimize:
            d_x_t = np.matmul(d_linear, Wx.T)
            d_h_prev = np.matmul(d_linear, Wh.T)
            self.grads['h_prev_grad'][...] = d_h_prev

        self.grads['Wx_grad'][...] = d_Wx
        self.grads['Wh_grad'][...] = d_Wh
        self.grads['bias_grad'][...] = d_bias

        return self.grads


class RNNLayer:
    def __init__(self, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        D, H = input_dim, hidden_dim
        self.input_dim, self.hidden_dim = D, H
        Wx = init.normal(D, H) if Wx is None else Wx
        Wh = init.normal(H, H) if Wh is None else Wh
        bias = init.normal(H) if bias is None else bias
        self.parameters = [Wx, Wh, bias]
        self.grads = {'Wx_grad': np.zeros_like(Wx),
                      'Wh_grad': np.zeros_like(Wh),
                      'bias_grad': np.zeros_like(bias)}
        self.timestep_cells = []

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
            h_stack[:, t, :] = h_next   # h_next는 (N, H)의 한 timestep rnn 결과물
            h_prev = h_next


        # T*N*D Style (Deprecated)
        # for time_step, x_t in enumerate(x_sequence):
        #     timestep_cell = RNNCell(self.N, self.D, self.hidden_dim, Wx=Wx, Wh=Wh, bias=bias)
        #     h_next = timestep_cell.forward(x_t, h_prev)
        #     self.rnn_steps.append(timestep_cell)
        #     h_prev = h_next

        h_last = h_next
        return h_last, h_stack

    def backward(self, d_h=1):
        _, Wx, Wh, bias = self.parameters
        d_Wx = np.zeros_like(Wx)
        d_Wh = np.zeros_like(Wh)
        d_bias = np.zeros_like(bias)

        for idx, layer in enumerate(reversed(self.timestep_cells)):
            grad = layer.backward(d_h_next=d_h)
            d_Wx += grad['Wx_grad']
            d_Wh += grad['Wh_grad']
            d_bias += grad['bias_grad']
            d_h = grad['h_prev_grad']

        for d in [d_Wx, d_Wh, d_bias]:
            np.clip(d, -1, 1, out=d)

        self.grads['Wx_grad'][...] = d_Wx
        self.grads['Wh_grad'][...] = d_Wh
        self.grads['bias_grad'][...] = d_bias
        self.grads["h_init_grad"][...] = d_h

        return self.grads
