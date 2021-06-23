import argparse
from matplotlib import pyplot as plt
from transformers import BartTokenizer
from transformers import AdamW
from sentence_transformers import SentenceTransformer
import torch
import torch.nn as nn
import numpy as np
import os
import tqdm
import sys

sys.path.insert(0, os.path.abspath('..'))
from interview_dataset import InterviewDatasetESPN
from model.BartWiki import BartWiki
from SBERT_filtering import find_top_k

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
MODEL_NAME = f'bart-base-wiki_bsz_{BATCH_SIZE}'
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.log'), 'w')

print(f'Training BART base with Wiki for {NUM_EPOCH} epochs, with batch size {BATCH_SIZE}')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
model = BartWiki.from_pretrained('facebook/bart-base').to(device)
sentence_encoder = SentenceTransformer('paraphrase-distilroberta-base-v1').to(device)
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

# optimizer
no_decay = ['bias', 'LayerNorm.weight']
optimizer_grouped_parameters = [
    {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
    {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
]
optimizer = AdamW(optimizer_grouped_parameters, lr=1e-5)

# load intermediate check point
# checkpoint = torch.load(os.path.join('model_weights', f'{MODEL_NAME}_checkpoint.tar'))
# model.load_state_dict(checkpoint['model_state_dict'])
# optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
# print("loaded")
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
    dataset = InterviewDatasetESPN(use_wiki=True)
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    # training
    train_iterator_with_progress = tqdm.tqdm(data_loader)
    idx = 0
    for batch in train_iterator_with_progress:
        
        try:

            batch_q = batch['question']
            batch_r = batch['response']
            batch_game_wiki = batch['game_wiki_id']
            batch_section_wiki = batch['section_wiki_id']
            batch_respondent_wiki = batch['respondent_wiki']
            for i in range(len(batch_q)):
                if batch_game_wiki[i] != '':
                    batch_game_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_game_wiki[i]))
                if batch_section_wiki[i] != '':
                    batch_section_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_section_wiki[i]))
                if batch_respondent_wiki[i] != '':
                    batch_respondent_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_respondent_wiki[i]))

            section_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_section_wiki))
            game_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_game_wiki))
            respondent_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_respondent_wiki))

            # input encoding
            input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True).to(device)

            # target encoding
            # this kind of embedding will make the input to BART decoder be like </s> <s> content </s>
            target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids'].to(device)
            target_ids[target_ids == model.config.pad_token_id] = -100

            # zero-out gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(**input_encoding,
                            labels=target_ids, 
                            section_wiki_encoding=section_wiki_encoding, 
                            game_wiki_encoding=game_wiki_encoding,
                            respondent_wiki_encoding=respondent_wiki_encoding)
            
            # backward pass
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # logging
            idx += 1
            total_loss += float(loss)
            train_iterator_with_progress.set_description(f'Epoch {epo}')
            train_iterator_with_progress.set_postfix({'Loss': loss.item()})
        except Exception as e:
            print(e)
            torch.save({'epoch': epo, 
                        'model_state_dict': model.state_dict(), 
                        'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join('model_weights', f'{MODEL_NAME}_checkpoint.tar'))
            print("check point saved")
            sys.exit()

    loss_record.append(total_loss)
    print(f'Loss in epoch {epo}: {total_loss}')
    log_file.write(f'Epoch:{epo} ')
    log_file.write(f'Loss:{total_loss} ')

    SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}_epoch_{epo+1}.pt')
    # save model after training for one epoch
    torch.save(model.state_dict(), SAVE_PATH)

    # evaluation
    model.eval()
    with torch.no_grad():
        '''
        DataLoader
        '''
        valid_dataset = InterviewDatasetESPN(data='dev', use_wiki=True)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True
        )

        batch_num = 0
        total_loss = 0
        for batch in valid_data_loader:
            batch_q = batch['question']
            batch_r = batch['response']
            batch_game_wiki = batch['game_wiki_id']
            batch_section_wiki = batch['section_wiki_id']
            batch_respondent_wiki = batch['respondent_wiki']
            for i in range(len(batch_q)):
                if batch_game_wiki[i] != '':
                    batch_game_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_game_wiki[i]))
                if batch_section_wiki[i] != '':
                    batch_section_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_section_wiki[i]))
                if batch_respondent_wiki[i] != '':
                    batch_respondent_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_respondent_wiki[i]))

            section_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_section_wiki))
            game_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_game_wiki))
            respondent_wiki_encoding = torch.tensor(sentence_encoder.encode(batch_respondent_wiki))

            # input encoding
            input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True).to(device)

            # target encoding
            # this kind of embedding will make the input to BART decoder be like </s> <s> content </s>
            target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
            target_ids = target_encoding['input_ids'].to(device)
            target_ids[target_ids == model.config.pad_token_id] = -100

            # zero-out gradient
            optimizer.zero_grad()

            # forward pass
            outputs = model(**input_encoding,
                            labels=target_ids, 
                            section_wiki_encoding=section_wiki_encoding, 
                            game_wiki_encoding=game_wiki_encoding,
                            respondent_wiki_encoding=respondent_wiki_encoding)

            # loss
            loss = outputs.loss
            total_loss += float(loss)
            batch_num += 1

        perplexity = np.exp(total_loss / batch_num)
        ppl_record.append(perplexity)
        print(f'Perplexity: {perplexity}')
        log_file.write(f'Perplexity:{perplexity}\n')

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
