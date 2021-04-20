import argparse
import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as opt
from tqdm import tqdm
from transformers import BartTokenizer, get_linear_schedule_with_warmup
sys.path.insert(0, os.path.abspath('..'))
from interview_dataset import InterviewDataset, InterviewDatasetAlternatives
from model.Seq2Seq import Seq2Seq


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
arg_parser.add_argument(
    '--max_grad_norm',
    type=float,
    default=1.0,
    help=f'Max gradient norm'
)
arg_parser.add_argument(
    '-s', '--speaker',
    action='store_true',
    help=f'Use speaker embedding'
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
os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
if args.speaker:
    MODEL_NAME = f'speaker_bsz_{BATCH_SIZE}'
else:
    MODEL_NAME = f'seq2seq_bsz_{BATCH_SIZE}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

if args.speaker:
    print(f'Training speaker model for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')
else:
    print(f'Training Seq2Seq model for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
if args.speaker:
    model = Seq2Seq(use_speaker=True).to(device)
else:
    model = Seq2Seq().to(device)

# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
optimizer
'''
optimizer = opt.Adam(model.parameters())

# num_training_steps = (int(len(dataset_train) / BATCH_SIZE) + 1) * NUM_EPOCH
# scheduler = get_linear_schedule_with_warmup(
#     optimizer,
#     num_warmup_steps=int(0.01*num_training_steps),
#     num_training_steps=num_training_steps
# )

# training loop
for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0

    '''
    DataLoader
    '''
    dataset = InterviewDataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # training
    train_iterator_with_progress = tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        # input encoding
        input_encoding = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids']
        input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

        # target encoding
        target_encoding = tokenizer(batch['response'], return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids']
        target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

        # zero-out gradient
        optimizer.zero_grad()

        # forward pass
        if args.speaker:
            speaker_id = [int(s.split('|')[1]) for s in batch['respondent']]
            speaker_id = torch.tensor(speaker_id, dtype=torch.long).to(device)

            outputs = model(x=input_ids, y=target_ids, speaker_id=speaker_id)
            # outputs.shape: (target_len, batch_size, vocab_size)
        else:
            outputs = model(x=input_ids, y=target_ids)  # outputs.shape: (target_len, batch_size, vocab_size)

        # prepare labels for cross entropy by removing the first time stamp (<s>)
        labels = target_ids[1:, :]  # shape: (target_len - 1, batch_size)
        labels = labels.reshape(-1).to(device)  # shape: ((target_len - 1) * batch_size)

        # prepare model predicts for cross entropy by removing the last timestamp and merge first two axes
        outputs = outputs[:-1, ...]  # shape: (target_len - 1, batch_size, vocab_size)
        outputs = outputs.reshape(-1, outputs.shape[-1]).to(device)
        # shape: ((target_len - 1) * batch_size, vocab_size)

        # compute loss and perform a step
        criterion = nn.CrossEntropyLoss(ignore_index=1)  # ignore padding index
        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)  # gradient clipping
        optimizer.step()
        # scheduler.step()

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

    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)

# close log file
log_file.close()
