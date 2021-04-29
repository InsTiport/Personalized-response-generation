import torch
import torch.nn as nn
from model.BeamSearch import greedy_search
from model.Seq2Seq import Encoder


class PointerGeneratorDecoder(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, embedding=None, hidden_size=1024, num_layers=4,
                 dropout=0.1):
        super(PointerGeneratorDecoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.dropout = nn.Dropout(p=dropout)

        # attention
        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()
        self.attention_concat = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)

        # pointer-generator
        self.sigmoid = nn.Sigmoid()
        self.h_weight = nn.Parameter(torch.zeros(1, hidden_size))
        self.s_weight = nn.Parameter(torch.zeros(1, hidden_size))
        self.x_weight = nn.Parameter(torch.zeros(1, embed_size))
        self.g_bias = nn.Parameter(torch.zeros(1))

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, h_i, c_i, encoder_output, attention_mask, train=True):
        """
        Generate the logits and current hidden state at one timestamp t

        Parameters
        ----------
        x : torch.Tensor (decoder_seq_len, batch_size) if train is True else (1, batch_size)
            The input batch of words

        h_i : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t - 1

        c_i : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t - 1

        encoder_output : torch.Tensor (encoder_seq_len, batch_size, hidden_size)
            hidden states in the final layer of the encoder

        attention_mask : torch.Tensor (encoder_seq_len, batch_size)
            boolean tensor indicating the position of padding

        train : bool
            If train is True, then will pass gold responses instead of one word

        Returns
        ---------
        logits : torch.Tensor (decoder_seq_len, batch_size, vocab_size) if train is True else (batch_size, vocab_size)
            Un-normalized logits

        h_i : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t (current timestamp)

        c_i : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t (current timestamp)

        p_gen : torch.Tensor (decoder_seq_len, batch_size)
            generation probability for pointer-generator

        weights : torch.Tensor (batch_size, encoder_seq_len, decoder_seq_len)
            attention weights
        """

        embed = self.dropout(self.embedding(x))  # embed.shape: (decoder_seq_len or 1, batch_size, embed_size)

        lstm_out, (h_o, c_o) = self.rnn(embed, (h_i, c_i))
        # lstm_out.shape: (decoder_seq_len or 1, batch_size, hidden_size) (h_t)

        # dot attention of Luong's style (Luong et al., 2015)
        if train:
            # need to reshape lstm_out_transformed to use bmm
            # lstm_out_transformed.shape : (batch_size, hidden_size, decoder_seq_len)
            lstm_out_transformed = lstm_out.permute(1, 2, 0)

            # need to reshape encoder_output to use bmm
            # encoder_output.shape: (batch_size, encoder_seq_len, hidden_size)
            encoder_output = encoder_output.permute(1, 0, 2)

            # attention score and weight
            attention_score = torch.bmm(encoder_output, lstm_out_transformed)
            # attention_score.shape: (batch_size, encoder_seq_len, decoder_seq_len)
            attention_score[attention_mask.transpose(0, 1)] = 1e-10
            weights = self.softmax(attention_score)  # shape: (batch_size, encoder_seq_len, decoder_seq_len)

            # weighted sum of encoder hidden states to get context c_t
            weighted_sum = torch.einsum('bnm,bnd->mbd', weights, encoder_output)
            # weighted_sum.shape: (decoder_seq_len, batch_size, hidden_size)

        else:
            # need to reshape lstm_out_transformed to use bmm
            # lstm_out_transformed.shape : (batch_size, hidden_size, 1)
            lstm_out_transformed = lstm_out.permute(1, 2, 0)

            # need to reshape encoder_output to use bmm
            # encoder_output.shape: (batch_size, encoder_seq_len, hidden_size)
            encoder_output = encoder_output.permute(1, 0, 2)

            # attention score and weight
            attention_score = torch.bmm(encoder_output, lstm_out_transformed)
            # attention_score.shape: (batch_size, encoder_seq_len, 1)
            attention_score[attention_mask.transpose(0, 1).unsqueeze(2)] = 1e-10
            weights = self.softmax(attention_score)  # shape: (batch_size, encoder_seq_len, 1)

            # weighted sum of encoder hidden states to get context c_t
            weighted_sum = torch.sum(encoder_output * weights, dim=1)  # shape: (batch_size, hidden_size)
            weighted_sum = weighted_sum.unsqueeze(0)  # shape: (1, batch_size, hidden_size)

        # generation probability
        p_gen = torch.zeros(weighted_sum.shape[0] * weighted_sum.shape[1]).to(weighted_sum.device)
        # shape: (decoder_seq_len * batch_size)

        for matrix, weight, size in zip(
                [weighted_sum, lstm_out, embed],
                [self.h_weight, self.s_weight, self.x_weight],
                [self.hidden_size, self.hidden_size, self.embed_size]
        ):
            '''
            weight.shape: (1, size), matrix.shape: (decoder_seq_len, batch_size, size)
            => (decoder_seq_len * batch_size)
            '''
            matrix_reshaped = matrix.reshape(-1, size).unsqueeze(-1)
            # shape: (decoder_seq_len * batch_size, size, 1)
            weight_reshaped = weight.repeat(matrix_reshaped.shape[0], 1).unsqueeze(1)
            # shape: (decoder_seq_len * batch_size, 1, size)
            p_gen += torch.bmm(weight_reshaped, matrix_reshaped).squeeze()

        # sigmoid
        p_gen = self.sigmoid(p_gen + self.g_bias)  # shape: (decoder_seq_len * batch_size)
        p_gen = p_gen.reshape(*(weighted_sum.shape[:2]))  # shape: (decoder_seq_len, batch_size)

        # concatenate context c_t with h_t to get (h_t)~
        lstm_out = self.tanh(self.attention_concat(torch.cat((lstm_out, weighted_sum), dim=2)))
        # lstm_out.shape: (decoder_seq_len, batch_size, hidden_size)

        logits = self.out(self.dropout(lstm_out)).squeeze()
        # shape: (batch_size, vocab_size) or (decoder_seq_len, batch_size, vocab_size) if train

        return logits, h_o, c_o, p_gen, weights


class PointerGenerator(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, hidden_size=1024, num_layers=2, dropout=0.3):
        super(PointerGenerator, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        '''
        Special token ids:
        <bos>: 0
        <pad>: 1
        <eos>: 2
        <unk>: 3
        '''
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size, padding_idx=1)

        self.encoder = Encoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            embedding=self.embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            padding_token_id=1
        )

        self.decoder = PointerGeneratorDecoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            embedding=self.embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x, y):
        """
        This method is used to train the model, hence it assumes the presence of gold responses (:y:)
        If you want to use the model in generation, use self.generate() instead

        Parameters
        ----------
        x : torch.Tensor (max_input_seq_len, batch_size)
            The input batch of questions

        y : torch.Tensor (max_output_seq_len, batch_size)
            The input batch of gold responses

        Returns
        ---------
        torch.Tensor (max_output_seq_len, batch_size, vocab_size)
            The predicted logits

        """
        self.encoder.train()
        self.decoder.train()

        # encoder pass
        encoder_output, attention_mask, h, c = self.encoder(x)  # use encoder hidden/cell states for decoder
        # encoder_output.shape: (max_input_seq_len, batch_size, hidden_size)
        # attention_mask.shape: (max_input_seq_len, batch_size)

        # decoder pass
        logits, _, _, p_gen, weights = self.decoder(y, h, c, encoder_output, attention_mask)
        # logits.shape: (decoder_seq_len, batch_size, vocab_size)
        # p_gen.shape: (decoder_seq_len, batch_size)
        # weights.shape: (batch_size, encoder_seq_len, decoder_seq_len)

        # reshape p_gen
        p_gen = p_gen.unsqueeze(-1)
        p_gen = p_gen.expand(-1, -1, self.vocab_size)  # shape: (decoder_seq_len, batch_size, vocab_size)

        '''
        compute logits with :p_gen: considered
        '''
        # first component
        logits = torch.softmax(logits, dim=-1) * p_gen

        # second component
        tmp_logits = torch.zeros_like(logits)

        x_transposed = x.transpose(0, 1)  # x_transposed.shape: (batch_size, encoder_seq_len)
        weights_transposed = weights.permute(2, 0, 1)  # shape: (decoder_seq_len, batch_size, encoder_seq_len)
        for idx, batch in enumerate(x_transposed):
            tmp_logits[:, idx, batch] += weights_transposed[:, idx]
            tmp_logits[:, idx] *= 1 - p_gen[:, idx, 0]

        # compute new logits and take log (to use nll loss during training)
        logits += tmp_logits
        logits = torch.log(logits)

        return logits

    def generate(self, x):
        """
        This method is used for conditional generation

        Parameters
        ----------
        x : torch.Tensor (max_input_seq_len, 1)
            The input batch of questions

        Returns
        ---------
        list
            The generated list of tokens

        """
        self.encoder.eval()
        self.decoder.eval()

        encoder_output, attention_mask, h, c = self.encoder(x)  # use encoder hidden/cell states for decoder
        # encoder_output.shape: (max_input_seq_len, 1, hidden_size)
        # attention_mask.shape: (max_input_seq_len, 1)

        # list of tokens
        generated_ids = greedy_search(self.decoder, encoder_output, attention_mask, h, c)

        return generated_ids
