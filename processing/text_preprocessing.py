import os
import csv
import json
import tqdm
from utils import tokenize, generate_vocab, generate_word_index_map, generate_indexed_sentences

# navigate to root directory in the project
os.chdir('../')

data_file = 'utterance.csv'

tokens_list = list()
line_count = 0
with open(os.path.join('data', 'csv', data_file), 'r') as f:
    reader = csv.reader(f)
    for row in reader:
        tokens_list.append(tokenize(row[-1]))
        if line_count % 1000 == 0 and line_count >= 1000:
            print(line_count)
        line_count += 1

vocab = generate_vocab(tokens_list)
word2idx, idx2word = generate_word_index_map(vocab)
indexed_sentences = generate_indexed_sentences(tokens_list, word2idx)

# store the vocabulary dictionary for later use
os.mkdir('data/vocab')
json.dump(word2idx, open(os.path.join('data', 'vocab', 'word2idx'), 'w'))
json.dump(idx2word, open(os.path.join('data', 'vocab', 'idx2word'), 'w'))

with open(os.path.join('data', 'csv', data_file), 'r') as file_in:
    reader = csv.reader(file_in)
    with open(os.path.join('data', 'csv', 'utterance_index.csv'), 'w') as file_out:
        writer = csv.writer(file_out)
        line = 0
        for row in reader:
            if line > 0: # don't want to change the titles
                row[-1] = indexed_sentences[line]
            writer.writerow(row)
            line += 1