from deepnp.functions import *
from deepnp.layers import *
import deepnp.initializers as init


class Encoder:
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        D, H, V = embedding_dim, hidden_dim, vocab_size

        embed_W = init.simplexavier(V, D)
        lstm_Wx = init.simplexavier(D, 4 * H)
        lstm_Wh = init.simplexavier(H, 4 * H)
        lstm_b = init.simplexavier(4 * H)

        self.embedding = EmbeddingTimesteps(embed_W)
        self.lstm = LSTMLayerTimesteps(D, H, lstm_Wx, lstm_Wh, lstm_b, stateful=False)

        self.params = self.embedding.params + self.lstm.parameters
        self.grads = self.embedding.grad + self.lstm.grads
        self.h_last, self.h_stack = None, None

    def forward(self, x_seq):
        x_emb = self.embedding.forward(x_seq)
        h_last, h_stack = self.lstm.forward(x_emb)
        self.h_stack = h_stack
        return h_last

    def backward(self, d_h_last):
        d_h_stack = np.zeros_like(self.h_stack)
        d_h_stack[:, -1, :] = d_h_last

        d_out = self.lstm.backward(d_h_stack)
        d_out = self.embedding.backward(d_out)
        return d_out


class Decoder:
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        D, H, V = embedding_dim, hidden_dim, vocab_size

        embed_W = init.simplexavier(V, D)
        lstm_Wx = init.simplexavier(D, 4 * H)
        lstm_Wh = init.simplexavier(H, 4 * H)
        lstm_b = init.simplexavier(4 * H)
        fc_W = init.simplexavier(H, V)
        fc_b = init.simplexavier(V)

        self.embedding = EmbeddingTimesteps(embed_W)
        self.lstm = LSTMLayerTimesteps(D, H, lstm_Wx, lstm_Wh, lstm_b, stateful=True)
        self.fc = FCLayerTimesteps(fc_W, fc_b)

        self.params = self.embedding.params + self.lstm.parameters + self.fc.params
        self.grads = self.embedding.grad + self.lstm.grads + self.fc.grads

    def forward(self, x_seq, h):
        self.lstm.set_state(h)
        x_emb = self.embedding.forward(x_seq)
        h_last, h_stack = self.lstm.forward(x_emb)
        out = self.fc.forward(h_stack)
        return out

    def backward(self, d_out):
        d_out = self.fc.backward(d_out)
        d_out = self.lstm.backward(d_out)
        d_out = self.embedding.backward(d_out)
        d_h = self.lstm.d_h_0
        return d_h

    def generate(self, h, start_id, sample_size):
        sampled = []
        sample_id = start_id
        self.lstm.set_state(h)

        for _ in range(sample_size):
            x = np.array(sample_id).reshape((1, 1))
            out = self.embedding.forward(x)
            h_last, h_stack = self.lstm.forward(out)
            score = self.fc.forward(h_stack)

            sample_id = np.argmax(score.flatten())
            sampled.append(int(sample_id))
            self.lstm.reset_timesteps()

        return sampled


class Seq2Seq:
    def __init__(self, embedding_dim, hidden_dim, vocab_size):
        D, H, V = embedding_dim, hidden_dim, vocab_size
        self.encoder = Encoder(D, H, V)
        self.decoder = Decoder(D, H, V)
        self.loss = SoftmaxWithLossLayerTimesteps()

        self.params = self.encoder.params + self.decoder.params
        self.grads = self.encoder.grads + self.decoder.grads

    def forward(self, x_seq, y_seq):
        # N, T
        decoder_x, decoder_y = y_seq[:, :-1], y_seq[:, 1:]

        self.reset_timesteps()
        h_last = self.encoder.forward(x_seq)
        out = self.decoder.forward(decoder_x, h_last)
        average_loss, num_acc = self.loss.forward(out, decoder_y)
        return average_loss

    def backward(self, d_out=1):
        d_out = self.loss.backward(d_out)
        d_h_last = self.decoder.backward(d_out)
        self.encoder.backward(d_h_last)

    def generate(self, xs, start_id, sample_size):
        self.reset_timesteps()
        h = self.encoder.forward(xs)
        sampled = self.decoder.generate(h, start_id, sample_size)
        self.reset_timesteps()
        return sampled

    def reset_timesteps(self):
        self.encoder.lstm.reset_timesteps()
        self.decoder.lstm.reset_timesteps()