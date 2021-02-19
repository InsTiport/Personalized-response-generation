import os
import csv
import json
import sys
import tqdm
sys.path.insert(0, os.path.abspath('..'))
from processing.utils import tokenize, generate_vocab, generate_word_index_map, generate_indexed_sentences
from scraping.scraper import ID_LOOKUP


'''
Run this script to re-generate vocabularies from all csv files. Will ERASE previous generated vocab files.
'''


# navigate to root directory in the project
os.chdir('../')

# dynamically find all sport categories available
sports_type = list(ID_LOOKUP.keys())
sports_type = [s for s in sports_type if os.path.exists(os.path.join('data', 'csv', f'{s}_utterance.csv'))]

print(f'Found {len(sports_type)} csv files to build vocabulary from.')
# process every *_utterance.csv to build vocabulary
vocab = set()
for sport in sports_type:
    with open(os.path.join('data', 'csv', f'{sport}_utterance.csv'), 'r') as f:
        print(f'Building vocabulary for {sport}_utterance.csv...')
        reader = csv.reader(f)
        # perform tokenization row (sentence) by row (sentence)
        for row in tqdm.tqdm(reader):
            vocab.update(tokenize(row[-1]))

        f.close()

# generate vocabulary
word2idx, idx2word = generate_word_index_map(vocab)

# display helpful info
print(f'Vocabulary size is {len(vocab)}')

# store the vocabulary dictionary for later use
os.mkdir('data/vocab', exist_ok=True)
json.dump(word2idx, open(os.path.join('data', 'vocab', 'word2idx'), 'w'))
json.dump(idx2word, open(os.path.join('data', 'vocab', 'idx2word'), 'w'))
