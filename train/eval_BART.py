import argparse
import os
import datasets
import numpy as np
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

# setup args
arg_parser = argparse.ArgumentParser()
arg_parser.add_argument(
    '--batch_size',
    type=int,
    default=3,
    help=f'Specify evaluation batch size'
)

arg_parser.add_argument(
    '-s', '--sampling',
    action='store_true',
    help=f'Whether to use sampling methods'
)

arg_parser.add_argument(
    '--num_beams',
    type=int,
    default=1,
    help=f'Beam search size, with 1 being greedy decoding'
)

arg_parser.add_argument(
    '--temperature',
    type=float,
    default=1.,
    help=f'Temperature for beam search'
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
DEVICE_ID = 0  # adjust this to use an unoccupied GPU
EVAL_BATCH_SIZE = args.batch_size
MODEL_NAME = f'bart-base_epoch_10_bsz_2_small_utterance'

# specifications
r'''MAX_LEN = default value: max length of model input'''
r'''MIN_LEN = default value: 10'''

# beam search specification (using early stopping)
use_beam = not args.sampling
num_beams = args.num_beams  # 1 means greedy decoding (no beam search)
temperature = args.temperature  # default: 1.

# sampling-based method specification (change use_beam to False if use these methods)
top_p = args.top_p  # default: 1.
top_k = args.top_k  # default: 50

# stick to 1 for now
num_return_sentences = 1


'''
logging 
'''
log_file = open(os.path.join('model', f'{MODEL_NAME}.ev'), 'a+')

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
train_iterator = BucketIterator(dataset=train_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)
test_iterator = BucketIterator(dataset=test_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
SAVE_PATH = os.path.join('model', f'{MODEL_NAME}.pt')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
try:
    model.load_state_dict(torch.load(SAVE_PATH))
except RuntimeError:
    model.load_state_dict(torch.load(SAVE_PATH, map_location='cuda:0'))
model.eval()
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
compute BLEU and perplexity in validation set
'''
metric_bleu = datasets.load_metric('sacrebleu')
metric_bertscore = datasets.load_metric('bertscore')
with torch.no_grad():
    batch_num = 0
    perplexity_sum = 0
    for batch in tqdm(valid_iterator):
        # FIXME for now, skip all invalid question-answer pairs (those having questions longer than 685)
        remove_idx = [i for i, q in enumerate(batch.q) if len(q) >= 685]
        batch_q = [q for i, q in enumerate(batch.q) if i not in remove_idx]
        batch_r = [r for i, r in enumerate(batch.r) if i not in remove_idx]
        assert len(batch_q) == len(batch_r)
        if len(batch_q) == 0:
            continue

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

        # record perplexity
        perplexity_sum += np.exp(loss.item())

        # generation
        if use_beam:
            model_res_ids = model.generate(
                input_ids,
                max_length=model.config.max_position_embeddings,
                num_beams=num_beams,
                temperature=temperature,
                num_return_sequences=num_return_sentences,
                early_stopping=True
            )
        else:
            model_res_ids = model.generate(
                input_ids,
                do_sample=True,
                max_length=model.config.max_position_embeddings,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sentences
            )

        # add generated responses and gold responses for future BLEU computation
        predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                       model_res_ids]
        references = [[r] for r in batch_r]
        metric_bleu.add_batch(predictions=predictions, references=references)
        metric_bertscore.add_batch(predictions=predictions, references=references)

        batch_num += 1

    # BLEU
    score_bleu = metric_bleu.compute()
    score_bertscore = metric_bertscore.compute(lang='en')
    # ppl
    perplexity = perplexity_sum / batch_num

    print('last batch: ')
    print(f'Questions: {batch_q}')
    print(f'Model predictions: {predictions}')
    print(f'Gold responses: {references}')

    print(f'Perplexity: {perplexity}')
    print(f'BLEU: {round(score_bleu["score"], 1)} out of {round(100., 1)}')
    print(f'BertScore: {score_bertscore["score"]}')
    # write results to file
    log_file.write(f'eval_bsz:{EVAL_BATCH_SIZE} ')
    log_file.write(f'use_beam_search:{use_beam} ')
    if use_beam:
        log_file.write(f'beam_size:{num_beams} ')
        log_file.write(f'temperature:{temperature} ')
    else:
        log_file.write(f'p:{top_p} ')
        log_file.write(f'k:{top_k} ')
    log_file.write(f'perplexity:{round(perplexity, 2)} ')
    log_file.write(f'BLEU:{round(score_bleu["score"], 1)}')
    log_file.write(f'BertScore:{score_bertscore["score"]}\n')
    log_file.close()

# # sample predictions which get full BLEU score
# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"],
#               ["foo bar foobar", "foo bar foobar"]]
# results = metric.compute(predictions=predictions, references=references)
# print(round(results["score"], 1))