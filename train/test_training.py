from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
import numpy as np
import os
import tqdm

os.chdir('../')

# control 
torch.manual_seed(0)
np.random.seed(0)
torch.cuda.set_device(3)

'''
hyper-parameter 
'''

BATCH_SIZE = 1
NUM_EPOCH = 5
SAVE_PATH = os.path.join('model', 'BART')


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
    # training
    model.train()
    per_batch = tqdm.tqdm(train_iterator)
    for batch in per_batch:
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

        per_batch.set_description(f'Epoch {epo}')
        per_batch.set_postfix({'Loss': loss.item()})

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
