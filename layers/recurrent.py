from deepnp.functions import *


class RNNCell:
    def __init__(self, batch_size, input_dim, hidden_dim, Wx=None, Wh=None, bias=None):
        Wx = Wx if Wx is not None else np.random.normal(0, 1, size=(input_dim, hidden_dim))
        Wh = Wh if Wh is not None else np.random.normal(0, 1, size=(hidden_dim, hidden_dim))
        bias = bias if bias is not None else np.random.normal(0, 1, size=hidden_dim)
        self.batch_size = batch_size
        self.parameters = [Wx, Wh, bias]
        self.grads = {
            'Wx_grad': np.zeros_like(Wx),
            'Wh_grad': np.zeros_like(Wh),
            'bias_grad': np.zeros_like(bias),
            'h_prev_grad': np.zeros(shape=(batch_size, hidden_dim)),
        }
        self.cache = {
            'x_t': None,
            'h_prev': None,
            'h_next': None
        }

    def forward(self, x_t, h_prev):
        Wx, Wh, bias = self.parameters
        linear = np.matmul(x_t, Wx) + np.matmul(h_prev, Wh) + bias
        h_next = np.tanh(linear)
        self.cache = {
            'x_t': x_t,
            'h_prev': h_prev,
            'h_next': h_next
        }
        return h_next

    def backward(self, d_h_next=1):
        Wx, Wh, bias = self.parameters
        x_t, h_prev, h_next = self.cache['x_t'], self.cache['h_prev'], self.cache['h_next']

        d_linear = d_h_next * (1 - np.square(h_next))  # element-wise dot product
        d_bias = np.sum(d_linear, axis=0, keepdims=False)
        d_x_t = np.matmul(d_linear, Wx.T)
        d_h_prev = np.matmul(d_linear, Wh.T)
        d_Wx = np.matmul(x_t.T, d_linear)
        d_Wh = np.matmul(h_prev.T, d_linear)

        self.grads['Wx_grad'][...] = d_Wx
        self.grads['Wh_grad'][...] = d_Wh
        self.grads['bias_grad'][...] = d_bias
        self.grads['h_prev_grad'][...] = d_h_prev

        return self.grads


class RNNLayer:
    def __init__(self, batch_size, input_dim, hidden_dim, Wx=None, Wh=None, bias=None, h_init=None):
        self.input_dim, self.hidden_dim, self.batch_size = input_dim, hidden_dim, batch_size
        Wx = Wx if Wx is not None else np.random.normal(0, 1, size=(input_dim, hidden_dim))
        Wh = Wh if Wh is not None else np.random.normal(0, 1, size=(hidden_dim, hidden_dim))
        bias = bias if bias is not None else np.random.normal(0, 1, size=hidden_dim)
        h_init = h_init if h_init is not None else np.zeros(shape=(batch_size, hidden_dim))
        self.parameters = [h_init, Wx, Wh, bias]
        self.grads = {
            'Wx_grad': np.zeros_like(Wx),
            'Wh_grad': np.zeros_like(Wh),
            'bias_grad': np.zeros_like(bias),
            'h_init_grad': np.zeros_like(h_init)
        }
        self.time_layers = []

    def forward(self, x_seq):
        h_next: np.ndarray

        h_init, Wx, Wh, bias = self.parameters
        h_prev = np.zeros(shape=(x_seq.shape[1], self.hidden_dim))

        for time_step, x_t in enumerate(x_seq):
            if time_step > 0:
                h_prev = h_next.copy()
            layer_t = RNNCell(self.batch_size, self.input_dim, self.hidden_dim, Wx=Wx, Wh=Wh, bias=bias)
            h_next = layer_t.forward(x_t, h_prev)
            self.time_layers.append(layer_t)

        h_last = h_next.copy()
        return h_last

    def backward(self, d_last_hidden=1):
        _, Wx, Wh, bias = self.parameters
        d_Wx = np.zeros_like(Wx)
        d_Wh = np.zeros_like(Wh)
        d_bias = np.zeros_like(bias)

        for idx, layer in enumerate(reversed(self.time_layers)):
            if idx == 0:
                grad = layer.backward(d_h_next=d_last_hidden)
            else:
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
