import argparse
import os
import numpy as np
import torch
from transformers import BartForConditionalGeneration, BartTokenizer

# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
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
CUDA
'''
# control randomness
torch.manual_seed(0)
np.random.seed(0)

# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU

'''
model and tokenizer
'''
# load model
SAVE_PATH = os.path.join('model', f'{MODEL_NAME}.pt')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
if not torch.cuda.is_available():
    model.load_state_dict(torch.load(SAVE_PATH, map_location='cpu'))
else:
    try:
        model.load_state_dict(torch.load(SAVE_PATH))
    except RuntimeError:
        model.load_state_dict(torch.load(SAVE_PATH, map_location='cuda:0'))
model.eval()

# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

football_path = os.path.join('data', 'football')
name_set = set()
for sub_dir in os.scandir(football_path):
    if sub_dir.is_dir:
        name = sub_dir.name.split(',')
        name.reverse()
        name = ' '.join([s.strip() for s in name])
        name_set.add(name)

for name in name_set:
    question = f'Q: What do you think of {name}?'

    # input encoding
    input_encoding = tokenizer(question, return_tensors='pt')
    input_ids = input_encoding['input_ids'].to(device)

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

    predictions = tokenizer.decode(model_res_ids[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(question)
    print('A: ' + predictions + '\n')
