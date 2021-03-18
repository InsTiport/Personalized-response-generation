import os
import sys
import csv
import tqdm

sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

sports_type = list(ID_LOOKUP.keys())

os.chdir('../')
csv_path = os.path.join('data', 'csv')
os.makedirs('user_corpus', exist_ok=True)

name2idx = dict()
index = 0
for sport in sports_type:
    with open(os.path.join(csv_path, sport + '_utterance.csv')) as file_in:
        reader = csv.reader(file_in)
        next(reader)
        for row in tqdm.tqdm(reader):
            if not row[-2].startswith('['):
                name = row[-2] + ' ' + sport
                name = name.replace('/', ' ')
                name = name.replace('\\', ' ')
                if name not in name2idx.keys():
                    name2idx[name] = index
                    index += 1
                with open(os.path.join('data', 'user_corpus', name), 'a') as file_out:
                    file_out.write(row[-1] + '\n')