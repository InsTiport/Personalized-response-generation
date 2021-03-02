from transformers import BartForConditionalGeneration, BartTokenizer, BartConfig
from transformers import AdamW
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
import os
import tqdm
import csv
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 1
num_epochs = 5

# load dataset
os.chdir('../')
question = RawField()
response = RawField()
fields = {'question': ('q', question), 'response': ('r', response)}
dataset = TabularDataset(path=os.path.join('data', 'csv', 'single_turn_utterance.csv'), format='csv', fields=fields)
train_set, test_set, valid_set = dataset.split([0.98, 0.01, 0.01])
train_iterator = BucketIterator(dataset=train_set, batch_size=BATCH_SIZE)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=BATCH_SIZE)

model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

for epo in range(num_epochs):
    pbar = tqdm.tqdm(train_iterator)
    for batch in pbar:
        input_encoding = tokenizer(batch.q, return_tensors='pt', padding=True, truncation=True)
        target_encoding = tokenizer(batch.r, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids'].to(device)
        target_ids = target_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        pbar.set_description(f'Epoch {epo}')
        pbar.set_postfix({'Loss': loss.item()})
    
    model.eval()
    with torch.no_grad():
        total_loss = 0
        token_num = 0
        for batch in valid_iterator:
            input_encoding = tokenizer(batch.q, return_tensors='pt', padding=True, truncation=True)
            target_encoding = tokenizer(batch.r, return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids'].to(device)
            target_ids = target_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)
            loss = outputs.loss
            total_loss += loss.item()
            token_num += torch.sum(torch.ones(target_ids.size(), device=device) * (target_ids != 0).to(device))

        perplexity = torch.exp(total_loss / token_num)
        print(f'Perplexity: {perplexity}')

    model.train()
