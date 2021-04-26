import torch
import torch.nn as nn
from model.BeamSearch import greedy_search


class Encoder(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, embedding=None, hidden_size=1024, num_layers=4,
                 dropout=0.1, padding_token_id=1):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.padding_token_id = padding_token_id
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor (max_seq_len, batch_size)
            The input batch of sentences

        Returns
        ---------
        encoder_output : torch.Tensor (max_seq_len, batch_size, hidden_size)
            encoder hidden states at each timestamp

        attention_mask : torch.Tensor (max_seq_len, batch_size)
            attention mask for encoder_output (1 means do not use attention)

        h_o : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t = max_seq_len

        c_o : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t = max_seq_len
        """

        # embedding layer
        embed = self.dropout(self.embedding(x))  # embed.shape = (max_seq_len, batch_size, embed_size)
        # lstm layer
        encoder_output, (h_o, c_o) = self.rnn(embed)

        # attention mask
        attention_mask = x == self.padding_token_id

        return encoder_output, attention_mask, h_o, c_o


class Decoder(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, embedding=None, hidden_size=1024, num_layers=4,
                 dropout=0.1, use_attention=True, use_speaker=False, num_speakers=25294, speaker_emb_dim=512):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.dropout = nn.Dropout(p=dropout)

        # attention
        self.use_attention = use_attention
        if use_attention:
            self.softmax = nn.Softmax(dim=1)
            self.tanh = nn.Tanh()
            self.attention_concat = nn.Linear(in_features=2*hidden_size, out_features=hidden_size)

        # speaker model
        self.use_speaker = use_speaker
        if use_speaker:
            self.num_speakers = num_speakers
            self.speaker_embed_dim = speaker_emb_dim

        if use_speaker:
            self.speaker_embedding = nn.Embedding(num_embeddings=num_speakers, embedding_dim=speaker_emb_dim)
            self.rnn = nn.LSTM(
                input_size=embed_size+speaker_emb_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )
        else:
            self.rnn = nn.LSTM(
                input_size=embed_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout
            )

        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, h_i, c_i, encoder_output, attention_mask, speaker_id=None, train=True):
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

        speaker_id : int (Optional)
            Id of speaker used in speaker model

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
        """

        embed = self.dropout(self.embedding(x))  # embed.shape: (decoder_seq_len or 1, batch_size, embed_size)

        if self.use_speaker:
            speaker_emb = self.dropout(self.speaker_embedding(speaker_id)).expand(x.shape[0], -1, -1)
            lstm_out, (h_o, c_o) = self.rnn(torch.cat((embed, speaker_emb), dim=2), (h_i, c_i))
            # lstm_out.shape: (decoder_seq_len or 1, batch_size, hidden_size) (h_t)
        else:
            lstm_out, (h_o, c_o) = self.rnn(embed, (h_i, c_i))
            # lstm_out.shape: (decoder_seq_len or 1, batch_size, hidden_size) (h_t)

        # dot attention of Luong's style (Luong et al., 2015)
        if self.use_attention:
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

                # concatenate context c_t with h_t to get (h_t)~
                lstm_out = self.tanh(self.attention_concat(torch.cat((lstm_out, weighted_sum), dim=2)))
                # lstm_out.shape: (decoder_seq_len, batch_size, hidden_size)
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

                # concatenate context c_t with h_t to get (h_t)~
                lstm_out = self.tanh(self.attention_concat(torch.cat((lstm_out, weighted_sum), dim=2)))
                # lstm_out.shape: (1, batch_size, hidden_size)

        logits = self.out(self.dropout(lstm_out)).squeeze()
        # shape: (batch_size, vocab_size) or (decoder_seq_len, batch_size, vocab_size) if train

        return logits, h_o, c_o


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, hidden_size=1024, num_layers=2, dropout=0.3,
                 use_attention=True, use_speaker=False, num_speakers=25294, speaker_emb_dim=256):
        super(Seq2Seq, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.use_attention = use_attention

        self.use_speaker = use_speaker
        if use_speaker:
            self.num_speakers = num_speakers
            self.speaker_emb_dim = speaker_emb_dim

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

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            embedding=self.embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            use_attention=use_attention,
            use_speaker=use_speaker,
            num_speakers=num_speakers,
            speaker_emb_dim=speaker_emb_dim
        )

    def forward(self, x, y, speaker_id=None):
        """
        This method is used to train the model, hence it assumes the presence of gold responses (:y:)
        If you want to use the model in generation, use self.generate() instead

        Parameters
        ----------
        x : torch.Tensor (max_input_seq_len, batch_size)
            The input batch of questions

        y : torch.Tensor (max_output_seq_len, batch_size)
            The input batch of gold responses

        speaker_id : int (Optional)
            Id of speaker used in speaker model

        Returns
        ---------
        torch.Tensor (max_output_seq_len, batch_size, vocab_size)
            The predicted logits

        """
        self.encoder.train()
        self.decoder.train()

        encoder_output, attention_mask, h, c = self.encoder(x)  # use encoder hidden/cell states for decoder
        # encoder_output.shape: (max_input_seq_len, batch_size, hidden_size)
        # attention_mask.shape: (max_input_seq_len, batch_size)

        logits, _, _ = self.decoder(y, h, c, encoder_output, attention_mask, speaker_id=speaker_id)

        return logits

    def generate(self, x, speaker_id=None):
        """
        This method is used for conditional generation

        Parameters
        ----------
        x : torch.Tensor (max_input_seq_len, 1)
            The input batch of questions

        speaker_id : int (Optional)
            Id of speaker used in speaker model

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
        generated_ids = greedy_search(self.decoder, encoder_output, attention_mask, h, c, speaker_id=speaker_id)

        return generated_ids
