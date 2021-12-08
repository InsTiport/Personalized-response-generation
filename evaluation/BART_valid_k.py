import argparse
import os
import sys
import torch
from tqdm import tqdm
from transformers import BartForConditionalGeneration, BartTokenizer

sys.path.insert(0, os.path.abspath('..'))

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
MODEL_NAME = f'bart-base_bsz_{args.batch_size}_epoch_{args.epoch}'

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
sample_results_file = open(os.path.join('evaluation_results', 'BART_interviewees.txt'), 'w', encoding='utf-8')


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

with torch.no_grad():
    with open(os.path.join('data', 'interviewee.csv')) as r:
        for line in tqdm.tqdm(r):
            line = line.rstrip()
            if '_' not in line[:line.index(',')]:
                continue
            interviewee_name = [line[:line.index('_')]]

            print(interviewee_name)

            # input encoding
            input_encoding = tokenizer(f'Who is {interviewee_name}?', return_tensors='pt', padding=True, truncation=True).to(device)

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
            prediction = [tokenizer.decode(g, skip_special_tokens=True) for g in model_res_ids]

            sample_results_file.write(prediction[0] + '\n')

            break

sample_results_file.close()
