import os
import sys
import tqdm
import csv
import json
from langdetect import detect, lang_detect_exception
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP
from processing.utils import generate_indexed_sentences, tokenize

# navigate to root directory in the project
os.chdir('../')

# dynamically find all sport categories available
sports_type = list(ID_LOOKUP.keys())
sports_type = [s for s in sports_type if os.path.exists(os.path.join('data', 'csv', f'{s}_utterance.csv'))]
sports_type = ['football']

# load vocabulary
word2idx = json.load(open(os.path.join('data', 'vocab', 'word2idx')))

# generate indexed versions of *_utterance.csv
for sport in sports_type:
    print(f'Generating indexed version of {sport}_utterance.csv...')
    with open(os.path.join('data', 'csv', f'{sport}_utterance.csv'), 'r') as file_in:
        reader = csv.reader(file_in)
        with open(os.path.join('data', 'csv', f'{sport}_utterance_indexed.csv'), 'w') as file_out:
            writer = csv.writer(file_out)
            line = 0
            for row in tqdm.tqdm(reader):
                if line > 0:  # don't want to change the titles

                    # check whether this utterance is in English
                    try:
                        is_en = detect(row[-1]) == 'en'
                        if not is_en:
                            print(row[-1])
                    except lang_detect_exception.LangDetectException:
                        is_en = False
                        print(row[-1])

                    # if this utterance is in English, convert it into indexed version
                    if is_en:
                        try:
                            row[-1] = generate_indexed_sentences([tokenize(row[-1])], word2idx)[0]
                        except KeyError:
                            print(tokenize(row[-1]))
                            for token in tokenize(row[-1]):
                                if token not in word2idx.keys():
                                    print(f'{token} not in vocabulary.')
                            exit(0)
                    # add value for new column
                    row.append(is_en)
                else:  # new column: if this utterance is in English
                    row.append('is_en')

                # write to file
                writer.writerow(row)
                line += 1

            file_out.close()
        file_in.close()
