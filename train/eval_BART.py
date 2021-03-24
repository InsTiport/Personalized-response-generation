import argparse
import os
import datasets
import numpy as np
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer
from torch.nn.functional import log_softmax

# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

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
DEVICE_ID = args.gpu  # adjust this to use an unoccupied GPU
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
    model.load_state_dict(torch.load(SAVE_PATH, map_location=device))
except RuntimeError:
    model.load_state_dict(torch.load(SAVE_PATH, map_location='cuda:0'))
model.eval()
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
compute BLEU and perplexity in validation set
'''
metric_bleu = datasets.load_metric('sacrebleu')
metric_BERTScore = datasets.load_metric('bertscore')
with torch.no_grad():
    batch_num = 0
    total_loss = 0
    total_ppl = 0  # record sum of ppl
    N = 0  # number of validation samples

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
        total_loss += float(loss)

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
        metric_BERTScore.add_batch(predictions=predictions, references=references)

        batch_num += 1

    # BLEU
    score_bleu = metric_bleu.compute()
    # BERTScore
    score_bert_score = metric_BERTScore.compute(lang='en')
    # ppl
    perplexity = np.exp(total_loss / batch_num)
    perplexity_2 = (total_ppl / N).item()

    # compare some predictions with gold responses
    print('last batch: ')
    batch_q = [q.replace('\u2011', '') for q in batch_q]
    predictions = [p.replace('\u2011', '') for p in predictions]
    references = [r[0].replace('\u2011', '') for r in references]
    print_len = EVAL_BATCH_SIZE // 2  # print these number of predictions
    for q, prediction, gold in zip(batch_q[:print_len], predictions[:print_len], references[:print_len]):
        print(f'Question: {q}')
        print(f'Model prediction: {prediction}')
        print(f'Gold: {gold}\n')
        
    print(f'Perplexity: {perplexity}')
    print(f'Perplexity 2: {perplexity_2}')
    print(f'BLEU: {round(score_bleu["score"], 1)} out of {round(100., 1)}')
    print(f'BertScore: {torch.mean(torch.tensor(score_bert_score["f1"]))}')
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
    log_file.write(f'perplexity 2:{round(perplexity_2, 2)} ')
    log_file.write(f'BLEU:{round(score_bleu["score"], 1)} ')
    log_file.write(f'BertScore:{torch.mean(torch.tensor(score_bert_score["f1"]))}\n')  # average F-1 BERTScore
    log_file.close()

# # sample predictions which get full BLEU score
# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"],
#               ["foo bar foobar", "foo bar foobar"]]
# results = metric.compute(predictions=predictions, references=references)
# print(round(results["score"], 1))
