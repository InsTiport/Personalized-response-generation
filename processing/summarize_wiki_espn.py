from logging import exception
import os
import argparse
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path

# setup args
arg_parser = argparse.ArgumentParser()

arg_parser.add_argument(
    '--gpu',
    type=int,
    default=0,
    help=f'Specify which gpu to use'
)

args = arg_parser.parse_args()

# load summarizer
summarizer = pipeline("summarization", device=args.gpu)

os.chdir('../')

# espn
espn_file_path = list()
for path in Path('data/espn').rglob('*'):
    # check if it is a file
    if len(str(path).split('/')) == 7:
        espn_file_path.append(path)

# wiki
wiki_file_path = [os.path.join('data', 'wiki', str(i)) for i in range(3594)]

# interviewee_wiki
interviewee_wiki_path = []
for f in os.listdir('data/interviewee_wiki'):
    interviewee_wiki_path.append(os.path.join('data', 'interviewee_wiki', f))

for path in tqdm(espn_file_path + wiki_file_path + interviewee_wiki_path):
    with open(path, 'r', encoding='utf-8') as f:
        text = f.read()
        
        summarized = ''
        if len(text.split(' ')) < 800:
            summarized = text
        else:
            for i in range(0, len(text.split(' ')) // 800, 2):
                current_block = [' '.join(text.split(' ')[i * 800: (i+1) * 800]), ' '.join(text.split(' ')[(i + 1) * 800: min((i+2) * 800, len(text.split(' ')))])]
                summarized_block = summarizer(current_block, truncation=True, max_length=50, min_length=30, do_sample=False)[0]['summary_text']
                summarized = summarized + ' ' + summarized_block[0] + ' ' + summarized_block[1]
        
        try:
            with open(str(path) + '_summarized', 'w', encoding='utf-8') as summarized_file:
                summarized_file.write(summarized)
        except Exception as e:
            print(e)
            print(path)