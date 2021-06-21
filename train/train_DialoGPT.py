import argparse
from matplotlib import pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AdamW
import torch
import numpy as np
import os
import tqdm
import sys

sys.path.insert(0, os.path.abspath('..'))
from interview_dataset import InterviewDatasetESPN

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
    default=1,
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
MODEL_NAME = f'dialogpt-small_bsz_{BATCH_SIZE}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

print(f'Training DialoGPT for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
model = AutoModelForCausalLM.from_pretrained('microsoft/DialoGPT-small').to(device)
# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('microsoft/DialoGPT-small')
# add pad token to tokenizer (GPT does not have it). These will be masked out
tokenizer.pad_token = tokenizer.eos_token

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# record these for every epoch
loss_record = []
ppl_record = []
# training loop
for epo in range(NUM_EPOCH):
    model.train()
    total_loss = 0

    '''
    DataLoader
    '''
    dataset = InterviewDatasetESPN()
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # training
    train_iterator_with_progress = tqdm.tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        batch_q = batch['question']  # list of strings with len = bsz
        batch_r = batch['response']  # list of strings with len = bsz
        # concatenate questions and responses together
        inputs = [q + r for q, r in zip(batch_q, batch_r)]

        # input encoding
        input_encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)

        # encoding for questions, needed for masking out question part
        question_encoding = tokenizer(
            batch_q,
            return_tensors='pt',
            padding='max_length',
            max_length=input_encoding['input_ids'].shape[-1]
        ).to(device)

        # labels, by masking out padding tokens and question part (exclude them while computing loss)
        labels = input_encoding['input_ids'].detach().clone()
        labels[labels == question_encoding['input_ids']] = -100

        # zero-out gradient
        optimizer.zero_grad()

        # forward pass
        outputs = model(**input_encoding, labels=labels)

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

    loss_record.append(total_loss)
    print(f'Loss in epoch {epo}: {total_loss}')
    log_file.write(f'Epoch:{epo} ')
    log_file.write(f'Loss:{total_loss} ')

    # evaluation
    model.eval()
    with torch.no_grad():
        '''
        DataLoader
        '''
        valid_dataset = InterviewDatasetESPN(data='dev')
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        batch_num = 0
        total_loss = 0
        for batch in valid_data_loader:
            batch_q = batch['question']  # list of strings with len = bsz
            batch_r = batch['response']  # list of strings with len = bsz
            # concatenate questions and responses together
            inputs = [q + r for q, r in zip(batch_q, batch_r)]

            # input encoding
            input_encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)

            # encoding for questions, needed for masking out question part
            question_encoding = tokenizer(
                batch_q,
                return_tensors='pt',
                padding='max_length',
                max_length=input_encoding['input_ids'].shape[-1]
            ).to(device)

            # labels, by masking out padding tokens and question part (exclude them while computing loss)
            labels = input_encoding['input_ids'].detach().clone()
            labels[labels == question_encoding['input_ids']] = -100

            # forward pass
            outputs = model(**input_encoding, labels=labels)

            # loss
            loss = outputs.loss
            total_loss += float(loss)
            batch_num += 1

        perplexity = np.exp(total_loss / batch_num)
        ppl_record.append(perplexity)
        print(f'Perplexity: {perplexity}')
        log_file.write(f'Perplexity:{perplexity}\n')

    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)

# close log file
log_file.close()

# plot loss and ppl
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

epochs = list(range(NUM_EPOCH))
ax[0].plot(epochs, loss_record)
ax[0].set_title('Loss', fontsize=20)
ax[0].set_xlabel('Epoch', fontsize=15)
ax[0].set_ylabel('Loss', fontsize=15)

ax[1].plot(epochs, ppl_record)
ax[1].set_title('Perplexity', fontsize=20)
ax[1].set_xlabel('Epoch', fontsize=15)
ax[1].set_ylabel('Perplexity', fontsize=15)

fig.savefig(os.path.join('figures', f'{MODEL_NAME}'))

# load model
# model = BartForConditionalGeneration()
# model.load_state_dict(torch.load(SAVE_PATH))
# model.eval()
