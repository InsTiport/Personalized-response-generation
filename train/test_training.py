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

# hyper-parameter
BATCH_SIZE = 1
NUM_EPOCH = 5
SAVE_PATH = os.path.join('model', 'BART')

# load dataset
question = RawField()
response = RawField()
fields = {'question': ('q', question), 'response': ('r', response)}
dataset = TabularDataset(path=os.path.join('data', 'csv', 'single_turn_utterance.csv'), format='csv', fields=fields)

# FIXME: Does it produce consistent splitting?
train_set, test_set, valid_set = dataset.split([0.98, 0.01, 0.01])
train_iterator = BucketIterator(dataset=train_set, batch_size=BATCH_SIZE)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=BATCH_SIZE)

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
        total_loss = 0
        token_num = 0
        for batch in valid_iterator:
            # input encoding
            input_encoding = tokenizer(batch.q, return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)

            # target encoding
            target_encoding = tokenizer(batch.r, return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids'].to(device)
            target_ids[target_ids == 1] = -100

            # forward pass
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

            # loss
            loss = outputs.loss
            total_loss += loss.item()
            token_num += torch.count_nonzero(attention_mask != 0)

        perplexity = torch.exp(total_loss / token_num)
        print(f'Perplexity: {perplexity}')

# save model
torch.save(model.state_dict(), SAVE_PATH)

# load model
# model = BartForConditionalGeneration()
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()
