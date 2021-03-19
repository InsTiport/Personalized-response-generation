import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchtext.data import RawField, TabularDataset, BucketIterator
import torch.optim as opt
from tqdm import tqdm
from transformers import BartTokenizer


class Encoder(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, embedding=None, hidden_size=1024, num_layers=4,
                 dropout=0.1):
        super(Encoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
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
        h_o : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t = max_seq_len

        c_o : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t = max_seq_len
        """

        # embedding layer
        embed = self.dropout(self.embedding(x))  # embed.shape = (max_seq_len, batch_size, embed_size)
        # lstm layer
        lstm_out, (h_o, c_o) = self.rnn(embed)

        return h_o, c_o


class Decoder(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, embedding=None, hidden_size=1024, num_layers=4,
                 dropout=0.1):
        super(Decoder, self).__init__()

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = embedding
        self.dropout = nn.Dropout(p=dropout)

        self.rnn = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

        self.out = nn.Linear(in_features=hidden_size, out_features=vocab_size)

    def forward(self, x, h_i, c_i):
        """
        Parameters
        ----------
        x : torch.Tensor (batch_size)
            The input batch of words

        h_i : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t - 1

        c_i : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t - 1

        Returns
        ---------
        out : torch.Tensor (batch_size, vocab_size)
            Logits

        h_i : torch.Tensor (num_layers, batch_size, hidden_size)
            hidden state of LSTM at t (current timestamp)

        c_i : torch.Tensor (num_layers, batch_size, hidden_size)
            cell state of LSTM at t (current timestamp)
        """
        # add extra dimension
        x = x.unsqueeze(0)  # x.shape: (1, batch_size)

        embed = self.dropout(self.embedding(x))  # embed.shape: (1, batch_size, embed_size)

        lstm_out, (h_o, c_o) = self.rnn(embed, (h_i, c_i))  # lstm_out.shape: (1, batch_size, hidden_size)

        out = self.out(lstm_out)  # out.shape: (1, batch_size, vocab_size)
        out = out.squeeze()  # out.shape: (batch_size, vocab_size)

        return out, h_o, c_o


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size=50265, embed_size=1024, hidden_size=1024, num_layers=2, dropout=0.1):
        super(Seq2Seq, self).__init__()

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
            dropout=dropout
        )

        self.decoder = Decoder(
            vocab_size=vocab_size,
            embed_size=embed_size,
            embedding=self.embedding,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout
        )

    def forward(self, x, y=None, train=False):
        """
        Parameters
        ----------
        x : torch.Tensor (max_input_seq_len, batch_size)
            The input batch of questions

        y : torch.Tensor (max_output_seq_len, batch_size)
            The input batch of gold responses

        train : bool
            If true, will use gold responses as inputs to the decoder

        Returns
        ---------
        if train:
            torch.Tensor (max_output_seq_len, batch_size, vocab_size)
                The predicted logits
        else:
            skip for now
        """
        bsz = x.shape[1]

        h, c = self.encoder(x)  # use encoder hidden/cell states for decoder

        if train:
            max_output_seq_len = y.shape[0]

            # will hold logits for output
            logits = torch.zeros(max_output_seq_len, bsz, self.vocab_size)

            for t in range(max_output_seq_len):
                # decoder inputs
                decoder_in = y[t]  # shape: (batch_size)

                # decode one step
                decoder_out, h, c = self.decoder(decoder_in, h, c)  # decoder_out.shape: (batch_size, vocab_size)

                # store logits
                logits[t] = decoder_out

            return logits
        else:
            pass


# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=5,
    help=f'Specify number of training epochs'
)
arg_parser.add_argument(
    '-b', '--batch',
    type=int,
    default=2,
    help=f'Specify batch size'
)
args = arg_parser.parse_args()
os.chdir('../')

'''
hyper-parameter 
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
BATCH_SIZE = args.batch
NUM_EPOCH = args.epoch

'''
control and logging
'''
# control randomness
torch.manual_seed(0)
np.random.seed(0)
# model saving and logging paths
MODEL_NAME = f'seq2seq_{NUM_EPOCH}_bsz_{BATCH_SIZE}_small_utterance'
os.makedirs(os.path.dirname('model' + '/'), exist_ok=True)
SAVE_PATH = os.path.join('model', f'{MODEL_NAME}.pt')
log_file = open(os.path.join('model', f'{MODEL_NAME}.log'), 'w')

print(f'Training sequence2sequence model for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

'''
load dataset
'''
# prepare fields (needed when loading dataset)
question = RawField()
response = RawField()
fields = {'question': ('q', question), 'response': ('r', response)}
# load dataset
train_set, valid_set, test_set = TabularDataset.splits(path=os.path.join('data', 'csv'),
                                                       train='smaller_utterance_train.csv',
                                                       validation='smaller_utterance_valid.csv',
                                                       test='smaller_utterance_test.csv',
                                                       format='csv',
                                                       fields=fields)

# # used for debugging
# train_set.examples = train_set.examples[:10]
# valid_set.examples = valid_set.examples[:10]

# split dataset into batches
train_iterator = BucketIterator(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = BucketIterator(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
model = Seq2Seq().to(device)
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
optimizer
'''
optimizer = opt.Adam(model.parameters())

# training loop
for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0

    # training
    train_iterator_with_progress = tqdm(train_iterator)
    idx = 0
    for batch in train_iterator_with_progress:
        # FIXME for now, skip all invalid question-answer pairs (those having questions longer than 685)
        remove_idx = [i for i, q in enumerate(batch.q) if len(q) >= 685]
        batch_q = [q.replace('\u2011', '') for i, q in enumerate(batch.q) if i not in remove_idx]
        batch_r = [r.replace('\u2011', '') for i, r in enumerate(batch.r) if i not in remove_idx]
        assert len(batch_q) == len(batch_r)
        if len(batch_q) == 0:
            continue

        # input encoding
        input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids']
        input_ids = torch.transpose(input_ids, 0, 1).to(device)  # need reshape to satisfy model input format

        # target encoding
        target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids']
        target_ids = torch.transpose(target_ids, 0, 1).to(device)  # need reshape to satisfy model input format

        # zero-out gradient
        optimizer.zero_grad()

        # forward pass
        outputs = model(x=input_ids, y=target_ids, train=True)  # outputs.shape: (target_len, batch_size, vocab_size)

        # prepare labels for cross entropy by removing the first time stamp (<s>)
        labels = target_ids[1:, :]  # shape: (target_len - 1, batch_size)
        labels = labels.reshape(-1).to(device)  # shape: ((target_len - 1) * batch_size)

        # prepare model predicts for cross entropy by removing the last timestamp and merge first two axes
        outputs = outputs[:-1, ...]  # shape: (target_len - 1, batch_size, vocab_size)
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(device)  # shape: ((target_len - 1) * batch_size, vocab_size)

        # compute loss and perform a step
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # if idx % 1000 == 0:
        #     print(f'epoch: {epo}, batch: {idx}, memory reserved {torch.cuda.memory_reserved(DEVICE_ID) / 1e9} GB')
        #     print(f'epoch: {epo}, batch: {idx}, memory allocated {torch.cuda.memory_allocated(DEVICE_ID) / 1e9} GB')
        idx += 1

        total_loss += float(loss)
        train_iterator_with_progress.set_description(f'Epoch {epo}')
        train_iterator_with_progress.set_postfix({'Loss': loss.item()})

    print(f'Loss in epoch {epo}: {total_loss}')
    log_file.write(f'Epoch:{epo} ')
    log_file.write(f'Loss:{total_loss}\n')


# save model
torch.save(model.state_dict(), SAVE_PATH)
# close log file
log_file.close()
