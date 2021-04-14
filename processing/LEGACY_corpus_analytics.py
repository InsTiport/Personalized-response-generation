import matplotlib.pyplot as plt
import csv
import os
import json
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

'''
Run this script to generate a overview of the dataset
'''

# navigate to root directory in the project
os.chdir('../')

# total number of utterances
utterances_count = 0
# total number of words
word_count = 0
# total number of episodes
episode_count = 0
players2num_utterances = dict()
for entry in os.scandir(os.path.join('data', 'csv')):
    if len(entry.name.split('.')[0].split('_')) == 2 and entry.name.split('.')[0].split('_')[1] == 'utterance':
        with open(entry.path, 'r') as file_in:
            reader = csv.reader(file_in)
            next(reader)
            for row in reader:
                utterances_count += 1
                if int(row[0]) > episode_count:
                    episode_count = int(row[0])
                if row[-2] != '[Q]' and row[-2] != '[MODERATOR]':
                    if row[-2] in players2num_utterances.keys():
                        players2num_utterances[row[-2]] += 1
                    else:
                        players2num_utterances[row[-2]] = 1
    elif len(entry.name.split('.')[0].split('_')) == 3 and entry.name.split('.')[0].split('_')[2] == 'indexed':
        with open(entry.path, 'r') as file_in:
            reader = csv.reader(file_in)
            next(reader)
            for row in reader:
                # utterance is of the form "[1, 2, 3, 4]"
                utterance = row[-1][1: len(row[-1]) - 1].split(',')
                word_count += len(utterance)

# sort the dict based on the number of utterances (from high to low)
players2num_utterances = dict(sorted(players2num_utterances.items(), key=lambda item: -item[1]))

# plot the number of utterances among all the players
fig, ax = plt.subplots()
ax.bar(range(len(players2num_utterances)), list(players2num_utterances.values()))
plt.yscale('log')
ax.set_xlabel('player')
ax.set_ylabel('number of utterances (log)')
plt.savefig(os.path.join('figures', 'num_utterances_log.png'))

# see the players with top 10 number of utterances
print("The players with top 10 number of utterances: ")
count = 0
for name, num in players2num_utterances.items():
    print(name, num)
    count += 1
    if count > 10:
        break

print("\n")
print(f"Vocabulary size: {len(json.load(open(os.path.join('data', 'vocab', 'word2idx'), 'r')))}")
print(f"Total number of utterances: {utterances_count}")
print(f"Total number of words: {word_count}")
print(f"Total number of episodes: {episode_count}")