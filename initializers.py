from typing import *
from deepnp.nploader import *


def normal(*args):
    return np.random.normal(0, 1, size=(*args,))


def zeros(*args):
    return np.zeros(shape=(*args,))


def empty(*args):
    return np.empty(shape=(*args,))
