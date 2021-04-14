import argparse
from torch.nn.functional import log_softmax
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers import AdamW
import torch
import numpy as np
import os
import tqdm
from interview_dataset import InterviewDataset

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
os.makedirs(os.path.dirname('model_weights' + '/'), exist_ok=True)
MODEL_NAME = f'bart-base_epoch_{NUM_EPOCH}_bsz_{BATCH_SIZE}'
SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}.pt')
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

print(f'Training for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

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

    '''
    DataLoader
    '''
    dataset = InterviewDataset()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    # training
    train_iterator_with_progress = tqdm.tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        batch_q = batch['question']
        batch_r = batch['response']

        # input encoding
        input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)

        # target encoding
        target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
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
        '''
        DataLoader
        '''
        valid_dataset = InterviewDataset(data='dev')
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=0
        )

        batch_num = 0
        perplexity_sum = 0
        total_loss = 0
        # used to perform macro-average ppl
        total_ppl = 0  # record sum of ppl
        N = 0  # number of validation samples
        for batch in valid_data_loader:
            batch_q = batch['question']
            batch_r = batch['response']

            # input encoding
            input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True)
            input_ids = input_encoding['input_ids'].to(device)
            attention_mask = input_encoding['attention_mask'].to(device)

            # target encoding
            target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
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

            N += len(batch_q)  # accumulate the number of validation samples from this batch
            predicted_prob = logits_flatten[range(logits_flatten.shape[0]), target_ids.view(-1)].view(target_ids.shape)
            predicted_prob[target_ids < 0] = 0  # zero out paddings, shape: (bsz, seq_len)
            ppl = - predicted_prob.sum(dim=1) / (target_ids >= 0).sum(dim=1)
            ppl = torch.exp(ppl)  # ppl for each individual sample in this batch
            total_ppl += ppl.sum()  # accumulate total ppl

        perplexity = np.exp(total_loss / batch_num)
        perplexity_2 = (total_ppl / N).item()
        print(f'Perplexity: {perplexity}')
        print(f'Perplexity (avg): {perplexity_2}')
        log_file.write(f'Perplexity:{perplexity} ')
        log_file.write(f'Perplexity_avg:{perplexity_2}\n')
        
# save model
torch.save(model.state_dict(), SAVE_PATH)
# close log file
log_file.close()

# load model
# model = BartForConditionalGeneration()
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()
