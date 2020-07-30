from deepnp.nploader import *


def softmax(x):
    if x.ndim == 1:
        x = x - x.max()
        x = np.exp(x)
        return x / x.sum()
    elif x.ndim == 2:
        x = x - x.max(axis=1, keepdims=True)
        x = np.exp(x)
        return x / x.sum(axis=1, keepdims=True)
    else:
        raise Exception("Not Supported Dimension")


def cross_entropy_error(y_pred, y_true):
    if y_pred.ndim == 1:
        y_true = y_true.reshape(1, y_true.size)
        y_pred = y_pred.reshape(1, y_pred.size)

    # one hot -> label index
    if y_pred.size == y_true.size:
        y_true = y_true.argmax(axis=1)

    batch_size = y_pred.shape[0]

    y_hat = y_pred[np.arange(batch_size), y_true]
    log_y_hat = np.log(y_hat + 1e-7)
    losses = -np.sum(log_y_hat)
    return losses / batch_size
