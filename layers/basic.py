from deepnp.functions import *
from deepnp.namedtuples import GradTuples


class SoftmaxLayer:
    def __init__(self):
        self.params, self.grads = [], []
        self.out = None

    def forward(self, x):
        self.out = softmax(x)
        return self.out

    def backward(self, dout):
        dx = self.out * dout
        sumdx = np.sum(dx, axis=1, keepdims=True)
        dx -= self.out * sumdx
        return dx


class SoftmaxWithLossLayer:
    def __init__(self):
        self.y_pred = None
        self.y_true = None

    def forward(self, x, y_true):
        self.y_true = y_true
        self.y_pred = softmax(x)
        loss, num_acc = cross_entropy_error(self.y_pred, self.y_true)
        return loss, num_acc

    def backward(self, d_out=1):
        batch_size = self.y_true.shape[0]

        dx = self.y_pred.copy()
        dx[np.arange(batch_size), self.y_true] -= 1
        dx *= d_out
        dx = dx / batch_size

        return dx


class FCLayer:
    def __init__(self, W, bias, batch_size):
        self.params = [W, bias]
        # self.grads = GradTuples.fc(W=np.zeros_like(W),
        #                            x=np.zeros(shape=(batch_size, W.shape[0]), dtype=float),
        #                            bias=np.zeros_like(bias))
        self.grads = {
            'W_grad': np.zeros_like(W),
            'x_grad': np.zeros(shape=(batch_size, W.shape[0]), dtype=float),
            'bias_grad': np.zeros_like(bias),
        }
        self.x = None

    def forward(self, x):
        self.x = x
        W, bias = self.params
        out = np.matmul(x, W) + bias
        return out

    def backward(self, dout):
        W, bias = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads['W_grad'][...] = dW
        self.grads['x_grad'][...] = dx
        self.grads['bias_grad'][...] = db

        return self.grads
