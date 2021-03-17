import argparse
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
import numpy as np
import os
import tqdm

# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=5,
    help=f'Specify number of training epochs'
)
arg_parser.add_argument(
    '-b', '--batch',
    type=int,
    default=3,
    help=f'Specify batch size'
)
args = arg_parser.parse_args()
os.chdir('../')

'''
hyper-parameter 
'''
DEVICE_ID = 3
BATCH_SIZE = args.batch
NUM_EPOCH = args.epoch
SAVE_PATH = os.path.join('model', f'bart-base_epoch_{NUM_EPOCH}_bsz_{BATCH_SIZE}_small_utterance')

print(f'Training for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

# control
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.set_device(DEVICE_ID)

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
# split dataset into batches
train_iterator = BucketIterator(dataset=train_set, batch_size=BATCH_SIZE, shuffle=True)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=BATCH_SIZE, shuffle=True)
test_iterator = BucketIterator(dataset=test_set, batch_size=BATCH_SIZE, shuffle=True)

'''
model and tokenizer
'''
# model and tokenizer
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
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
    train_iterator_with_progress = tqdm.tqdm(train_iterator)
    idx = 0
    for batch in train_iterator_with_progress:
        # FIXME for now, skip all invalid question-answer pairs
        for q in batch.q:
            if len(q) >= 685:
                continue

        # input encoding
        input_encoding = tokenizer(batch.q, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)

        # target encoding
        target_encoding = tokenizer(batch.r, return_tensors='pt', padding=True, truncation=True)
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

        if idx % 1000 == 0:
            print(f'epoch: {epo}, batch: {idx}, memory reserved {torch.cuda.memory_reserved(DEVICE_ID) / 1e9} GB')
            print(f'epoch: {epo}, batch: {idx}, memory allocated {torch.cuda.memory_allocated(DEVICE_ID) / 1e9} GB')
        idx += 1

        total_loss += float(loss)
        train_iterator_with_progress.set_description(f'Epoch {epo}')
        train_iterator_with_progress.set_postfix({'Loss': loss.item()})

    print(f'Loss in epoch {epo}: {total_loss}')

    # evaluation
    model.eval()
    with torch.no_grad():
        batch_num = 0
        perplexity_sum = 0
        for batch in valid_iterator:
            # FIXME for now, skip all invalid question-answer pairs
            for q in batch.q:
                if len(q) >= 685:
                    continue

            # input encoding
            input_encoding = tokenizer(batch.q, return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)

            # target encoding
            target_encoding = tokenizer(batch.r, return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids'].to(device)
            target_ids[target_ids == model.config.pad_token_id] = -100

            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

            # loss
            loss = outputs.loss
            perplexity_sum += np.exp(loss.item())
            batch_num += 1

        perplexity = perplexity_sum / batch_num
        print(f'Perplexity: {perplexity}')

# save model
torch.save(model.state_dict(), SAVE_PATH)

# load model
# model = BartForConditionalGeneration()
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()
