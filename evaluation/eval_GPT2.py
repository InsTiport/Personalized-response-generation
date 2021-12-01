import argparse
import os
import sys
import datasets
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import pipeline

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
    default=1,
    help=f'Specify evaluation batch size'
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
log_file = open(os.path.join('evaluation_results', 'gpt2.ev'), 'a+')
sample_results_file = open(os.path.join('evaluation_results', 'gpt2_sample_results.txt'), 'w', encoding='utf-8')

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
    print("yes")

# load model
model = AutoModelForCausalLM.from_pretrained('gpt2').to(device)
model.eval()

# load tokenizer
tokenizer = AutoTokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token

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

    # load summarizer
    summarizer = pipeline("summarization", device=DEVICE_ID)

    batch_num = 0
    total_loss = 0
    distinct_one = []
    distinct_two = []
    prediction_len = []
    for batch in tqdm(test_data_loader):
        # create prompt
        batch_prompt = list() # list of strings with len = bsz
        batch_final_prompt = list()
        for b in range(EVAL_BATCH_SIZE):
            prompt = ''
            interviewee_name = batch['respondent'][b].split('|')[0].split('_')[0]
            if batch['game_wiki_id'][b] != '':
                with open(os.path.join('data', 'wiki', batch['game_wiki_id'][b]), 'r') as wiki_file:
                    wiki_text = wiki_file.read()
                    summarized = ''
                    for i in range(len(wiki_text.split(' ')) // 800):
                        current_block = ' '.join(wiki_text.split(' ')[i * 800: (i+1) * 800])
                        summarized_block = summarizer(current_block, truncation=True, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
                        summarized = summarized + ' ' + summarized_block
                    prompt = prompt + 'Here\'s the information for the game. ' + summarized
            if batch['section_wiki_id'][b] != '':
                with open(os.path.join('data', 'wiki', batch['section_wiki_id'][b]), 'r') as wiki_file:
                    wiki_text = wiki_file.read()
                    summarized = ''
                    for i in range(len(wiki_text.split(' ')) // 800):
                        current_block = ' '.join(wiki_text.split(' ')[i * 800: (i+1) * 800])
                        summarized_block = summarizer(current_block, truncation=True, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
                        summarized = summarized + ' ' + summarized_block
                    prompt = prompt + 'Here\'s the information for the section. ' + wiki_file.read()
            if batch['espn'][b] != '':
                espn_text = batch['espn'][b]
                summarized = ''
                for i in range(len(espn_text.split(' ')) // 800):
                    if i > 10: break
                    current_block = ' '.join(espn_text.split(' ')[i * 800: (i+1) * 800])
                    summarized_block = summarizer(current_block, truncation=True, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
                    summarized = summarized + ' ' + summarized_block
                prompt = prompt + 'Here\'s a report for the game. ' + summarized
            if batch['respondent_wiki'][b] != '':
                with open(batch['respondent_wiki'][b], 'r') as wiki_file:
                    wiki_text = wiki_file.read()
                    summarized = ''
                    for i in range(len(wiki_text.split(' ')) // 800):
                        current_block = ' '.join(wiki_text.split(' ')[i * 800: (i+1) * 800])
                        summarized_block = summarizer(current_block, truncation=True, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
                        summarized = summarized + ' ' + summarized_block
                    prompt = prompt + f'Here\'s information of {interviewee_name}. ' + summarized
            # if batch['prev_response'][b] != '':
            #     prompt = prompt + 'Here\'s the previous response in the interview' + batch['prev_response'][b]
            final_prompt = f'. The sports reporter asks {interviewee_name}:"'
            batch_final_prompt.append(final_prompt)
            prompt = prompt + final_prompt

            batch_prompt.append(prompt)
        
        # groundtruth questions
        batch_question = batch['question']

        # concatenate questions and responses together
        inputs = [p + q for p, q in zip(batch_prompt, batch_question)]

        # input encoding
        input_encoding = tokenizer(inputs, return_tensors='pt', padding=True, truncation=True).to(device)

        # encoding for questions, needed for masking out question part
        prompt_encoding = tokenizer(
            batch_prompt,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=input_encoding['input_ids'].shape[-1]
        ).to(device)

        # labels, by masking out padding tokens and question part (exclude them while computing loss)
        labels = input_encoding['input_ids'].detach().clone()
        try:
            labels[labels == prompt_encoding['input_ids']] = -100
        except RuntimeError:
            print(f'labels.shape: {labels.shape}')
            print(f"prompt_encoding.shape: {prompt_encoding['input_ids'].shape}")
            print(f"input_encoding.shape: {input_encoding['input_ids'].shape}")
            continue

        # forward pass
        outputs = model(**input_encoding, labels=labels)

        # loss
        loss = outputs.loss
        total_loss += float(loss)
        batch_num += 1

        input_encoding = tokenizer(
            batch_prompt,
            return_tensors='pt',
            padding=True,
            truncation=True,
        ).to(device)

        # generation
        if use_beam:
            model_res_ids = model.generate(
                input_encoding['input_ids'],
                pad_token_id=tokenizer.eos_token_id,
                max_length=min(input_encoding['input_ids'].shape[-1] + 128, 1024),
                num_beams=num_beams,
                early_stopping=True,
                num_return_sequences=num_return_sentences
            )
        else:
            model_res_ids = model.generate(
                input_encoding['input_ids'],
                pad_token_id=tokenizer.eos_token_id,
                max_length=min(input_encoding['input_ids'].shape[-1] + 128, 1024),
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                num_return_sequences=num_return_sentences
            )

        # add generated questions and gold questions for future BLEU computation
        predictions = [tokenizer.decode(g, skip_special_tokens=True) for g in model_res_ids]
        predictions = [prediction[prediction.find(final_prompt) + len(final_prompt):] for prediction, final_prompt in zip(predictions, batch_final_prompt)]
        predictions = [prediction[:prediction.find('\"')] for prediction in predictions]

        tmp_predictions, tmp_questions = [], []
        for prediction, question in zip(predictions, batch['question']):
            if len(question) > 0:
                tmp_predictions.append(prediction)
                tmp_questions.append(question)
        predictions, responses = tmp_predictions, tmp_questions

        for prediction in predictions:
            distinct_one.append(distinct_n_sentence_level(prediction, 1))
            distinct_two.append(distinct_n_sentence_level(prediction, 2))
            prediction_len.append(len(prediction.split()))

        references = [[r] for r in responses]
        metric_bleu.add_batch(predictions=predictions, references=references)
        metric_BERTScore.add_batch(predictions=predictions, references=references)

        # record sample
        # if np.random.choice([True, False], p=[0.1, 0.9]):
        predictions = [p.replace('\u2011', '') for p in predictions]
        references = [r[0].replace('\u2011', '') for r in references]
        for prediction, gold in zip(predictions, references):
            try:
                sample_results_file.write(f'Model prediction: {prediction}\n')
                sample_results_file.write(f'Gold: {gold}\n\n')
                print(prediction)
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
predictions = [p.replace('\u2011', '') for p in predictions]
references = [r[0].replace('\u2011', '') for r in references]
print_len = EVAL_BATCH_SIZE // 2  # print these number of predictions
print_len = EVAL_BATCH_SIZE
for prediction, gold in zip(predictions[:print_len], references[:print_len]):
    print(f'Model prediction: {prediction}')
    print(f'Gold: {gold}\n')