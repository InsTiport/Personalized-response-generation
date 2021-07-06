import argparse
import os
import sys
import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

sys.path.insert(0, os.path.abspath('..'))
from interview_dataset import InterviewDatasetESPN
from metrics.distinct_n import distinct_n_sentence_level

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

arg_parser.add_argument(
    '--num_beams',
    type=int,
    default=1,
    help=f'Beam search size, with 1 being greedy decoding'
)

arg_parser.add_argument(
    '-s', '--sampling',
    action='store_true',
    help=f'Whether to use sampling methods'
)

arg_parser.add_argument(
    '--temperature',
    type=float,
    default=1.,
    help=f'Temperature'
)

arg_parser.add_argument(
    '--top_k',
    type=int,
    default=50,
    help=f'Top-k sampling'
)

arg_parser.add_argument(
    '--top_p',
    type=float,
    default=1.,
    help=f'Top-p nucleus sampling'
)

args = arg_parser.parse_args()
os.chdir('../')

'''
hyper-parameter and generation specifications
'''
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
EVAL_BATCH_SIZE = args.eval_batch_size
MODEL_NAME = f'bart-base-prev-concat_bsz_{args.batch_size}_epoch_{args.epoch}'

# specifications
r'''MAX_LEN = default value: max length of model input'''
r'''MIN_LEN = default value: 10'''

# beam search specification (using early stopping)
use_beam = not args.sampling
num_beams = args.num_beams  # 1 means greedy decoding (no beam search)

# sampling-based method specification (change use_beam to False if use these methods)
temperature = args.temperature  # default: 1
top_p = args.top_p  # default: 1.
top_k = args.top_k  # default: 50

# stick to 1 for now
num_return_sentences = 1


'''
logging 
'''
os.makedirs(os.path.dirname('evaluation_results' + '/'), exist_ok=True)
log_file = open(os.path.join('evaluation_results', f'{MODEL_NAME}.ev'), 'a+')
sample_results_file = open(os.path.join('evaluation_results', f'{MODEL_NAME}_sample_results.txt'), 'w', encoding='utf-8')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU

# load model
SAVE_PATH = os.path.join('model_weights', f'{MODEL_NAME}.pt')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
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
    test_dataset = InterviewDatasetESPN(data='test')
    test_data_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=EVAL_BATCH_SIZE
    )

    batch_num = 0
    total_loss = 0
    distinct_one = []
    distinct_two = []
    prediction_len = []
    for batch in tqdm(test_data_loader):
        batch_q = batch['question']
        batch_r = batch['response']
        batch_prev_q = batch['prev_question']
        batch_prev_r = batch['prev_response']

        inputs = [prev_q + tokenizer.sep_token + prev_r + tokenizer.sep_token + q
                  for q, prev_q, prev_r in zip(batch_q, batch_prev_q, batch_prev_r)]

        # input encoding
        input_encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)

        # target encoding
        # this kind of embedding will make the input to BART decoder be like </s> <s> content </s>
        target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids'].to(device)
        target_ids[target_ids == model.config.pad_token_id] = -100

        # forward pass
        outputs = model(**input_encoding, labels=target_ids)

        # loss
        loss = outputs.loss
        total_loss += float(loss)

        batch_num += 1

        # generation
        if use_beam:
            model_res_ids = model.generate(
                input_encoding['input_ids'],
                max_length=model.config.max_position_embeddings,
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=num_return_sentences
            )
        else:
            model_res_ids = model.generate(
                input_encoding['input_ids'],
                max_length=model.config.max_position_embeddings,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sentences
            )

        # add generated responses and gold responses for future BLEU computation
        predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in model_res_ids]

        tmp_predictions, tmp_responses = [], []
        for prediction, response in zip(predictions, batch['response']):
            if len(response) > 0:
                tmp_predictions.append(prediction)
                tmp_responses.append(response)
        predictions, responses = tmp_predictions, tmp_responses

        for prediction in predictions:
            distinct_one.append(distinct_n_sentence_level(prediction, 1))
            distinct_two.append(distinct_n_sentence_level(prediction, 2))
            prediction_len.append(len(prediction.split()))

        references = [[r] for r in responses]
        metric_bleu.add_batch(predictions=predictions, references=references)
        metric_BERTScore.add_batch(predictions=predictions, references=references)

        # record sample
        if np.random.choice([True, False], p=[0.1, 0.9]):
            batch_q = [q.replace('\u2011', '') for q in batch_q]
            predictions = [p.replace('\u2011', '') for p in predictions]
            references = [r[0].replace('\u2011', '') for r in references]
            batch_bg = [bg.replace('\u2011', '') for bg in batch_bg]
            for q, prediction, gold, bg in zip(batch_q, predictions, references, batch_bg):
                try:
                    sample_results_file.write(f'Background: {bg}\n')
                    sample_results_file.write(f'Question: {q}\n')
                    sample_results_file.write(f'Model prediction: {prediction}\n')
                    sample_results_file.write(f'Gold: {gold}\n\n')
                except Exception as e:
                    print(e)
        
    sample_results_file.close()

    # BLEU
    score_bleu = metric_bleu.compute()
    # BERTScore
    score_bert_score = metric_BERTScore.compute(lang='en')
    # ppl
    perplexity = np.exp(total_loss / batch_num)
        
    print(f'Perplexity: {perplexity}')
    print(f'BLEU: {round(score_bleu["score"], 1)} out of {round(100., 1)}')
    print(f'BertScore: {torch.mean(torch.tensor(score_bert_score["f1"]))}')
    print(f'Distinct-1: {torch.mean(torch.tensor(distinct_one))}')
    print(f'Distinct-2: {torch.mean(torch.tensor(distinct_two))}')
    print(f'Average length: {torch.mean(torch.tensor(prediction_len, dtype=torch.float))}')
    # write results to file
    log_file.write(f'eval_bsz:{EVAL_BATCH_SIZE} ')
    log_file.write(f'use_beam_search:{use_beam} ')
    if use_beam:
        log_file.write(f'beam_size:{num_beams} ')
    else:
        log_file.write(f'temperature:{temperature} ')
        log_file.write(f'p:{top_p} ')
        log_file.write(f'k:{top_k} ')
    log_file.write(f'perplexity:{round(perplexity, 2)} ')
    log_file.write(f'BLEU:{round(score_bleu["score"], 1)} ')
    log_file.write(f'BertScore:{torch.mean(torch.tensor(score_bert_score["f1"]))} ')  # average F-1 of BERTScore
    log_file.write(f'Distinct1:{torch.mean(torch.tensor(distinct_one))} ')
    log_file.write(f'Distinct2:{torch.mean(torch.tensor(distinct_two))} ')
    log_file.write(f'Avg_len:{torch.mean(torch.tensor(prediction_len, dtype=torch.float))}\n')
    log_file.close()

# # sample predictions which get full BLEU score
# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"],
#               ["foo bar foobar", "foo bar foobar"]]
# results = metric.compute(predictions=predictions, references=references)
# print(round(results["score"], 1))

# compare some predictions with gold responses
print('last batch: ')
batch_q = [q.replace('\u2011', '') for q in batch_q]
predictions = [p.replace('\u2011', '') for p in predictions]
references = [r[0].replace('\u2011', '') for r in references]
print_len = EVAL_BATCH_SIZE // 2  # print these number of predictions
print_len = EVAL_BATCH_SIZE
for q, prediction, gold in zip(batch_q[:print_len], predictions[:print_len], references[:print_len]):
    print(f'Question: {q}')
    print(f'Model prediction: {prediction}')
    print(f'Gold: {gold}\n')
