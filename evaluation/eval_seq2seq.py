import argparse
import os
import sys
import datasets
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from transformers import BartTokenizer
sys.path.insert(0, os.path.abspath('..'))
from interview_dataset import InterviewDataset
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
    '--eval_batch_size',
    type=int,
    default=3,
    help=f'Specify evaluation batch size'
)

arg_parser.add_argument(
    '-b', '--batch_size',
    type=int,
    default=5,
    help='Specify the batch size of the model while it was trained'
)

arg_parser.add_argument(
    '-e', '--epoch',
    type=int,
    default=3,
    help='Specify which epoch\'s checkpoint to use'
)

# arg_parser.add_argument(
#     '--num_beams',
#     type=int,
#     default=1,
#     help=f'Beam search size, with 1 being greedy decoding'
# )
#
# arg_parser.add_argument(
#     '-s', '--sampling',
#     action='store_true',
#     help=f'Whether to use sampling methods'
# )
#
# arg_parser.add_argument(
#     '--temperature',
#     type=float,
#     default=1.,
#     help=f'Temperature for beam search'
# )
#
# arg_parser.add_argument(
#     '--top_k',
#     type=int,
#     default=50,
#     help=f'Top-k sampling'
# )
#
# arg_parser.add_argument(
#     '--top_p',
#     type=float,
#     default=1.,
#     help=f'Top-p nucleus sampling'
# )

args = arg_parser.parse_args()
os.chdir('../')

'''
hyper-parameter and generation specifications
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
EVAL_BATCH_SIZE = args.eval_batch_size
MODEL_NAME = f'seq2seq_bsz_{args.batch_size}_epoch_{args.epoch}'

# # specifications
# r'''MAX_LEN = default value: max length of model input'''
# r'''MIN_LEN = default value: 10'''
#
# # beam search specification (using early stopping)
# use_beam = not args.sampling
# num_beams = args.num_beams  # 1 means greedy decoding (no beam search)
#
# # sampling-based method specification (change use_beam to False if use these methods)
# temperature = args.temperature  # default: 1
# top_p = args.top_p  # default: 1.
# top_k = args.top_k  # default: 50
#
# # stick to 1 for now
# num_return_sentences = 1


'''
logging 
'''
log_file = open(os.path.join('model_weights', f'{MODEL_NAME}.ev'), 'a+')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU

# load model
SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}.pt')
model = Seq2Seq().to(device)
model.load_state_dict(torch.load(SAVE_PATH, map_location=device))

model.eval()

# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
compute BLEU and perplexity in validation set
'''
metric_bleu = datasets.load_metric('sacrebleu')
metric_BERTScore = datasets.load_metric('bertscore')
with torch.no_grad():

    '''
    DataLoader
    '''
    test_dataset = InterviewDataset(data='test')
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE,
        shuffle=True
    )

    batch_num = 0
    total_loss = 0
    for batch in tqdm(test_data_loader):
        # input encoding
        input_encoding = tokenizer(batch['question'], return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids']
        input_ids = torch.transpose(input_ids, 0, 1).to(device)  # shape: (input_len, batch_size)

        # target encoding
        target_encoding = tokenizer(batch['response'], return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids']
        target_ids = torch.transpose(target_ids, 0, 1).to(device)  # shape: (target_len, batch_size)

        # # forward pass
        # if args.speaker:
        #     speaker_id = [int(s.split('|')[1]) for s in batch['respondent']]
        #     speaker_id = torch.tensor(speaker_id, dtype=torch.long).to(device)
        #
        #     outputs = model(x=input_ids, y=target_ids, speaker_id=speaker_id)
        #     # outputs.shape: (target_len, batch_size, vocab_size)
        # else:
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

        total_loss += float(loss)
        batch_num += 1

        # generation
        input_ids = torch.transpose(input_ids, 0, 1)  # shape: (batch_size, max_question_len)
        model_res_ids = []
        for question in input_ids:
            model_res_ids.append(model.generate(question))

        # add generated responses and gold responses for future BLEU computation
        predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in model_res_ids]
        references = [[r] for r in batch['response']]
        metric_bleu.add_batch(predictions=predictions, references=references)
        metric_BERTScore.add_batch(predictions=predictions, references=references)

    # BLEU
    score_bleu = metric_bleu.compute()
    # BERTScore
    score_bert_score = metric_BERTScore.compute(lang='en')
    # ppl
    perplexity = np.exp(total_loss / batch_num)
        
    print(f'Perplexity: {perplexity}')
    print(f'BLEU: {round(score_bleu["score"], 1)} out of {round(100., 1)}')
    print(f'BertScore: {torch.mean(torch.tensor(score_bert_score["f1"]))}')
    # write results to file
    log_file.write(f'eval_bsz:{EVAL_BATCH_SIZE} ')
    log_file.write(f'perplexity:{round(perplexity, 2)} ')
    log_file.write(f'BLEU:{round(score_bleu["score"], 1)} ')
    log_file.write(f'BertScore:{torch.mean(torch.tensor(score_bert_score["f1"]))}\n')  # average F-1 of BERTScore
    log_file.close()

# # sample predictions which get full BLEU score
# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"],
#               ["foo bar foobar", "foo bar foobar"]]
# results = metric.compute(predictions=predictions, references=references)
# print(round(results["score"], 1))

# compare some predictions with gold responses
print('last batch: ')
batch_q = [q.replace('\u2011', '') for q in batch['question']]
predictions = [p.replace('\u2011', '') for p in predictions]
references = [r[0].replace('\u2011', '') for r in references]
print_len = EVAL_BATCH_SIZE // 2  # print these number of predictions
for q, prediction, gold in zip(batch_q[:print_len], predictions[:print_len], references[:print_len]):
    print(f'Question: {q}')
    print(f'Model prediction: {prediction}')
    print(f'Gold: {gold}\n')
