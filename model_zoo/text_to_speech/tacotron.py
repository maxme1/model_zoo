from dpipe.layers import *


class Highway(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.h = nn.Sequential(
            nn.Linear(features, features),
            nn.ReLU(),
        )
        self.t = nn.Sequential(
            nn.Linear(features, features),
            nn.Sigmoid(),
        )

    def forward(self, x):
        w = self.t(x)
        return self.h(x) * w + x * (1 - w)


class CBHG(nn.Module):
    def __init__(self, in_features, filters, projections):
        super().__init__()

        hidden_features = in_features
        self.convs = nn.ModuleList([
            nn.Sequential(
                Lambda(functional.pad, pad=[(k - 1) // 2, k // 2]),
                PostActivation1d(in_features, f, kernel_size=k),
            ) for k, f in enumerate(filters, 1)
        ])
        self.fixed_convs = nn.Sequential(
            Lambda(functional.pad, pad=[1, 0]),
            nn.MaxPool1d(2, stride=1),

            make_consistent_seq(PostActivation1d, [sum(filters), *projections], kernel_size=3, padding=1)
        )
        self.highway = nn.Sequential(
            nn.Linear(projections[-1], hidden_features),
            *(Highway(hidden_features) for _ in range(4)),
        )

        self.bi_rnn = nn.GRU(hidden_features, hidden_features, bidirectional=True, batch_first=True)

    def forward(self, x):
        # batch_size, seq_len, n_features
        in_ = x
        x = x.transpose(-1, -2)
        x = torch.cat([conv(x) for conv in self.convs], 1)
        x = self.fixed_convs(x)

        x = in_ + x.transpose(-1, -2)
        x = self.highway(x)
        return self.bi_rnn(x)[0]


class AlignmentNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.query_net = nn.Linear(n_features, n_features, bias=False)
        self.aggregate = nn.Sequential(
            nn.Tanh(),
            nn.Linear(n_features, 1, bias=False),
        )

    def forward(self, query, embeddings):
        query = self.query_net(query)
        return self.aggregate(embeddings + query[:, None]).squeeze(-1)


def pre_net(in_features, out_features):
    return nn.Sequential(
        nn.Linear(in_features, in_features),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Dropout(0.5),
    )


class Decoder(nn.Module):
    def __init__(self, in_features, rnn_features, out_features):
        super().__init__()
        self.input_len = out_features
        self.pre_net = pre_net(out_features, in_features)
        self.attention_rnn = nn.GRUCell(in_features + in_features, in_features)
        self.alignment_net = AlignmentNet(in_features)
        self.to_decoder = nn.Linear(in_features * 2, rnn_features)
        self.to_mel = nn.Linear(rnn_features, out_features)

        self.decoder_rnn = nn.ModuleList([
            nn.GRUCell(rnn_features, rnn_features) for _ in range(2)
        ])

    def forward(self, embeddings, inputs=None, n_iterations=None):
        n_iterations = n_iterations or inputs.shape[1]

        context = torch.zeros_like(embeddings[:, 0])
        attention_state = None
        decoder_states = [None] * len(self.decoder_rnn)
        outputs = [torch.zeros(len(embeddings), self.input_len).to(context)]

        for step in range(n_iterations):
            inp = inputs[:, step] if inputs is not None else outputs[-1]
            inp = self.pre_net(inp)

            attention_state = self.attention_rnn(
                torch.cat([inp, context], 1),
                attention_state,
            )

            weights = self.alignment_net(attention_state, embeddings)
            weights = functional.softmax(weights, 1)

            context = (embeddings * weights[:, :, None]).sum(1)

            output = torch.cat([attention_state, context], 1)
            output = self.to_decoder(output)
            for idx, cell in enumerate(self.decoder_rnn):
                decoder_states[idx] = cell_output = cell(output, decoder_states[idx])
                output = output + cell_output

            output = self.to_mel(output)
            outputs.append(output)

        return torch.stack(outputs[1:], 1)


class Tacotron(nn.Module):
    def __init__(self, vocabulary_size, embedding_dim, hidden_features, encoder_filters, encoder_projections,
                 rnn_features, n_mels, decoder_filters, decoder_projections, output_features):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Embedding(vocabulary_size, embedding_dim),
            pre_net(embedding_dim, hidden_features),
            CBHG(hidden_features, encoder_filters, encoder_projections)
        )

        self.decoder = Decoder(hidden_features * 2, rnn_features, n_mels)
        self.post_net = nn.Sequential(
            CBHG(n_mels, decoder_filters, decoder_projections),
            nn.Linear(n_mels * 2, output_features),
        )

    def forward(self, x, inputs=None, n_iterations=None):
        embeddings = self.encoder(x)
        mels = self.decoder(embeddings, inputs, n_iterations)
        outputs = self.post_net(mels)
        return mels, outputs
