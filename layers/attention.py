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


class AttentionScoreDot:
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
        ht_r = np.repeat(ht_r, repeats=T, axis=1)  # N, T, H
        element_prod = ht_r * h_stack  # N, T, H
        dot = np.sum(element_prod, axis=2)  # N, T
        score = self.softmax.forward(dot)  # N, T

        self.cache = (h_stack, ht_r,)
        return score

    def backward(self, d_score):
        # d_score는 N, T
        h_stack, ht_r = self.cache
        N, T, H = self.shapes

        d_dot = self.softmax.backward(d_score)  # N, T
        d_element_prod = np.repeat(d_dot.reshape(N, T, 1), repeats=H, axis=2)  # N, T, H
        d_h_stack = d_element_prod * ht_r  # N, T, H
        d_ht_r = d_element_prod * h_stack  # N, T, H
        d_ht = np.sum(d_ht_r, axis=1)  # N, H

        return d_h_stack, d_ht


class AttentionCell:
    def __init__(self):
        self.params, self.grads = [], []
        self.attention_score_layer = AttentionScoreDot()
        self.weighted_sum_layer = WeightedSum()
        self.attention_score = None

    def forward(self, h_stack, ht):
        attention_score = self.attention_score_layer.forward(h_stack, ht)
        context = self.weighted_sum_layer.forward(h_stack, attention_score)
        self.attention_score = attention_score
        return context

    def backward(self, d_context):
        d_h_stack0, d_score = self.weighted_sum_layer.backward(d_context)
        d_h_stack1, d_ht = self.attention_score_layer.backward(d_score)
        d_h_stack = d_h_stack0 + d_h_stack1
        return d_h_stack, d_ht


class AttentionWithTimesteps:
    def __init__(self):
        self.params, self.grads = [], []
        self.timestep_layers = None
        self.attention_scores = None

    def forward(self, enc_h_stack, dec_h_stack):
        # 둘 다 N, T, H지만 encoder와 decoder의 T는 다를 수 있음
        N, T, H = dec_h_stack.shape  # T는 decoder의 timesteps
        contexts = np.empty_like(dec_h_stack)  # N, T, H 의 context vector
        self.initialize()

        for t in range(T):
            cell = AttentionCell()
            contexts[:, t, :] = cell.forward(enc_h_stack, dec_h_stack[:, t, :])
            self.timestep_layers.append(cell)
            self.attention_scores.append(cell.attention_score)

        return contexts

    def backward(self, d_contexts):
        N, T, H = d_contexts.shape  # T는 decoder의 timesteps
        d_enc_h_stack = 0
        d_dec_h_stack = np.empty_like(d_contexts)

        for t in range(T):  # reverse 안하는 이유 : t마다 개별적
            cell = self.timestep_layers[t]
            d_enc_h_stack_t, d_dec_ht = cell.backward(d_contexts[:, t, :])
            d_enc_h_stack += d_enc_h_stack_t
            d_dec_h_stack[:, t, :] = d_dec_ht

        return d_enc_h_stack, d_dec_h_stack

    def initialize(self):
        self.timestep_layers = list()
        self.attention_scores = list()
