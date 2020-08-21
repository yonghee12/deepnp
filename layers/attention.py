from . import *


class WeightedSum:
    def __init__(self):
        self.params, self.grads = [], []
        self.cache = None
        self.shapes = (0, 0, 0,)

    def forward(self, h_stack, weight):
        # h_stack은 (N, T, H), weight은 (N, T)
        assert weight.ndim == 2 and h_stack.ndim == 3
        N, T, H = h_stack.shape
        self.shapes = (N, T, H,)

        weight_r = weight.reshape(N, T, 1)  # w: N, T, 1
        weight_r = np.repeat(weight_r, repeats=H, axis=2)  # w: N, T, H
        h_weighted = weight_r * h_stack  # w: N, T, H
        context = h_weighted.sum(axis=1)  # w: N, H
        self.cache = (h_stack, weight_r,)  # w: N, T, H
        return context

    def backward(self, d_context):
        h_stack, weight_r = self.cache
        N, T, H = self.shapes
        d_context = d_context.reshape(N, 1, H)  # N, H -> N, 1, H
        d_h_weighted = np.repeat(d_context, repeats=T, axis=1)  # N, T, H
        d_h_stack = d_h_weighted * weight_r
        d_weight_r = d_h_weighted * h_stack  # N, T, H
        d_weight = np.sum(d_weight_r, axis=2)  # N, T
        return d_h_stack, d_weight


class DotProductScore:
    def __init__(self):
        self.params, self.grads = [], []
        self.softmax = SoftmaxLayer()
        self.cache = None
        self.shapes = (0, 0, 0,)

    def forward(self, h_stack, ht):
        # h_stack은 (N, T, H), ht는 (N, H)
        N, T, H = h_stack.shape
        self.shapes = (N, T, H,)

        ht_r = ht.reshape(N, 1, H)
        ht_r = np.repeat(ht_r, repeats=T, axis=1)   # N, T, H
        element_prod = ht_r * h_stack               # N, T, H
        dot = np.sum(element_prod, axis=2)          # N, T
        score = self.softmax.forward(dot)           # N, T

        self.cache = (h_stack, ht_r,)
        return score

    def backward(self, d_score):
        # d_score는 N, T
        h_stack, ht_r = self.cache
        N, T, H = self.shapes

        d_dot = self.softmax.backward(d_score)      # N, T
        d_element_prod = np.repeat(d_dot.reshape(N, T, 1), repeats=H, axis=2)   # N, T, H
        d_h_stack = d_element_prod * ht_r           # N, T, H
        d_ht_r = d_element_prod * h_stack           # N, T, H
        d_ht = np.sum(d_ht_r, axis=1)               # N, H

        return d_h_stack, d_ht
