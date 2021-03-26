import argparse
from torch.nn.functional import log_softmax
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
import numpy as np
import os
import tqdm
import pandas as pd

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
MODEL_NAME = f'bart-base_epoch_{NUM_EPOCH}_bsz_{BATCH_SIZE}_ConvAI2'
os.makedirs(os.path.dirname('model' + '/'), exist_ok=True)
SAVE_PATH = os.path.join('model', f'{MODEL_NAME}.pt')
log_file = open(os.path.join('model', f'{MODEL_NAME}.log'), 'w')

print(f'Training for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE} on ConvAI2')

'''
load dataset
'''
# # prepare fields (needed when loading dataset)
# question = RawField()
# response = RawField()
# fields = {'question': ('q', question), 'response': ('r', response)}
# # load dataset
# train_set, valid_set, test_set = TabularDataset.splits(path=os.path.join('data', 'csv'),
#                                                        train='smaller_utterance_train.csv',
#                                                        validation='smaller_utterance_valid.csv',
#                                                        test='smaller_utterance_test.csv',
#                                                        format='csv',
#                                                        fields=fields)
#
# # # used for debugging
# # train_set.examples = train_set.examples[:10]
# # valid_set.examples = valid_set.examples[:10]
#
# # split dataset into batches
# train_iterator = BucketIterator(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
# valid_iterator = BucketIterator(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True)
# test_iterator = BucketIterator(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

training_dataset = []
training_data = open(os.path.join('data', 'convai2_fix_723', 'train_both_original_no_cands.txt'), 'r')
conv = {
    # 'your_persona': [],
    # 'partner_persona': [],
    'conv': {
        'speaker_1': [],
        'speaker_2': []
    }
}  # used to store one conversation at a time
for line in training_data:
    line = line.strip().resplace(' .', '.').replace(' ?', '?').replace(' !', '!').replace(' ,', ',').resplace(' ;', ';')
    if line[:2] == '1 ':
        if len(conv['conv']['speaker_1']) > 0:
            training_dataset.append(conv)  # add finished conv to dataset
        conv = {
            # 'your_persona': [],
            # 'partner_persona': [],
            'conv': {
                'speaker_1': [],
                'speaker_2': []
            }
        }  # used to store one conversation at a time
        # conv['your_persona'].append(line[line.index(':') + 2:])
    else:
        # if 'your persona:' in line:
        #     conv['your_persona'].append(line[line.index(':') + 2:])
        # elif 'partner\'s persona:' in line:
        #     conv['partner_persona'].append(line[line.index(':') + 2:])
        # else:
        #     conv['conv'].append(line[line.index(' ') + 1:])
        if '\t' in line:
            conv['conv']['speaker_1'].append(line[line.index(' ') + 1:line.index('\t')])
            conv['conv']['speaker_2'].append(line[line.index('\t') + 1:])

training_dataset.append(conv)  # don't forget the last conversation
training_data.close()

validation_dataset = []
validation_data = open(os.path.join('data', 'convai2_fix_723', 'valid_both_original_no_cands.txt'), 'r')
conv = {
    # 'your_persona': [],
    # 'partner_persona': [],
    'conv': {
        'speaker_1': [],
        'speaker_2': []
    }
}  # used to store one conversation at a time
for line in validation_data:
    line = line.strip()
    if line[:2] == '1 ':
        if len(conv['conv']['speaker_1']) > 0:
            validation_dataset.append(conv)  # add finished conv to dataset
        conv = {
            # 'your_persona': [],
            # 'partner_persona': [],
            'conv': {
                'speaker_1': [],
                'speaker_2': []
            }
        }  # used to store one conversation at a time
        # conv['your_persona'].append(line[line.index(':') + 2:])
    else:
        # if 'your persona:' in line:
        #     conv['your_persona'].append(line[line.index(':') + 2:])
        # elif 'partner\'s persona:' in line:
        #     conv['partner_persona'].append(line[line.index(':') + 2:])
        # else:
        #     conv['conv'].append(line[line.index(' ') + 1:])
        if '\t' in line:
            conv['conv']['speaker_1'].append(line[line.index(' ') + 1:line.index('\t')])
            conv['conv']['speaker_2'].append(line[line.index('\t') + 1:])

validation_dataset.append(conv)  # don't forget the last conversation
validation_data.close()

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# training loop
for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0

    # training
    train_iterator_with_progress = tqdm.tqdm(training_dataset)
    idx = 0
    for batch in train_iterator_with_progress:
        # input encoding
        input_encoding = tokenizer(batch['conv']['speaker_1'], return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)

        # target encoding
        target_encoding = tokenizer(batch['conv']['speaker_2'], return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids'].to(device)
        target_ids[target_ids == model.config.pad_token_id] = -100

        # zero-out gradient
        optimizer.zero_grad()

        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

        # compute loss and perform a step
        loss = outputs.loss
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
    log_file.write(f'Loss:{total_loss} ')

    # evaluation
    model.eval()
    with torch.no_grad():
        batch_num = 0
        perplexity_sum = 0
        total_loss = 0
        # used to perform macro-average ppl
        total_ppl = 0  # record sum of ppl
        N = 0  # number of validation samples
        for batch in tqdm.tqdm(validation_dataset):
            # input encoding
            input_encoding = tokenizer(batch['conv']['speaker_2'], return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)

            # target encoding
            target_encoding = tokenizer(batch['conv']['speaker_2'], return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids'].to(device)
            target_ids[target_ids == model.config.pad_token_id] = -100

            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

            # loss
            loss = outputs.loss
            total_loss += float(loss)
            batch_num += 1

            # get logits and use it to calculate perplexity
            logits = outputs.logits  # shape: (bsz, seq_len, vocab_sz)
            logits = log_softmax(logits, dim=-1)
            logits_flatten = logits.view(-1, model.config.vocab_size)  # shape: (bsz * seq_len, vocab_sz)

            N += len(batch['conv']['speaker_1'])  # accumulate the number of validation samples from this batch
            predicted_prob = logits_flatten[range(logits_flatten.shape[0]), target_ids.view(-1)].view(target_ids.shape)
            predicted_prob[target_ids < 0] = 0  # zero out paddings, shape: (bsz, seq_len)
            ppl = - predicted_prob.sum(dim=1) / (target_ids >= 0).sum(dim=1)
            ppl = torch.exp(ppl)  # ppl for each individual sample in this batch
            total_ppl += ppl.sum()  # accumulate total ppl

        perplexity = np.exp(total_loss / batch_num)
        perplexity_2 = (total_ppl / N).item()
        print(f'Perplexity: {perplexity}')
        print(f'Perplexity (avg): {perplexity_2}')
        log_file.write(f'Perplexity:{perplexity}\n')

# save model
torch.save(model.state_dict(), SAVE_PATH)
# close log file
log_file.close()

# load model
# model = BartForConditionalGeneration()
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()
