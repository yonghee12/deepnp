from .recurrent import *
from .attention import *
from .. import initializers as init


class AttentionEncoder(Encoder):
    """
    기존 encoder와 다른 점: h_last뿐 아니라 h_stack도 같이 넘겨줌
    """
    def forward(self, x_seq):
        x_emb = self.embedding.forward(x_seq)
        h_last, h_stack = self.lstm.forward(x_emb)
        return h_last, h_stack

    def backward(self, d_h_stack):
        d_emb = self.lstm.backward(d_h_stack)
        d_x_seq = self.embedding.backward(d_emb)
        return d_x_seq


class AttentionDecoder:
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        D, H, V = embedding_dim, hidden_dim, vocab_size
        self.embeeding_dim, self.hidden_dim, self.vocab_size = D, H, V

        embed_W = init.simplexavier(V, D)
        lstm_Wx = init.simplexavier(D, 4 * H)
        lstm_Wh = init.simplexavier(H, 4 * H)
        lstm_b = init.simplexavier(4 * H)
        fc_W = init.simplexavier(2 * H, V)
        fc_b = init.simplexavier(V)

        self.embedding = EmbeddingTimesteps(embed_W)
        self.lstm = LSTMLayerTimesteps(D, H, lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.attention = AttentionLayerWithTimesteps()
        self.fc = FCLayerTimesteps(fc_W, fc_b)

        layers = [self.embedding, self.lstm, self.attention, self.fc]
        self.params, self.grads = [], []
        for layer in layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, x_seq, enc_h_stack, enc_h_last):
        self.lstm.set_state(enc_h_last)

        x_emb = self.embedding.forward(x_seq)
        dec_h_last, dec_h_stack = self.lstm.forward(x_emb)                      # N, T, H
        context = self.attention.forward(enc_h_stack, dec_h_stack)              # N, T, H
        h_context = np.concatenate(arrays=(context, dec_h_stack), axis=2)       # N, T, 2H
        out = self.fc.forward(h_context)   # N, T, V

        return out

    def backward(self, d_out):
        d_h_context = self.fc.backward(d_out)
        N, T, H2 = d_h_context.shape
        H = int(H2 / 2)

        d_context, d_dec_h_stack0 = d_h_context[:, :, :H], d_h_context[:, :, H:]
        