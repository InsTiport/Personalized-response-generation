import matplotlib.pyplot as plt
import csv
import os
import sys
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

'''
Run this script to generate a overview of the dataset
'''

# navigate to root directory in the project
os.chdir('../')

num_utterances = dict()
for entry in os.scandir(os.path.join('data', 'csv')):
    if entry.name.split('.')[0].split('_')[1] == 'utterance':
        with open(entry.path, 'r') as file_in:
            reader = csv.reader(file_in)
            for row in reader:
                if row[-2] != '[Q]' and row[-2] != '[MODERATOR]':
                    if row[-2] in num_utterances.keys():
                        num_utterances[row[-2]] += 1
                    else:
                        num_utterances[row[-2]] = 1
num_utterances = dict(sorted(num_utterances.items(), key=lambda item: -item[1]))

# plot the number of utterances among all the players
fig, ax = plt.subplots()
ax.bar(range(len(num_utterances)), list(num_utterances.values()))
plt.yscale('log')
ax.set_xlabel('player')
ax.set_ylabel('number of utterances (log)')
plt.savefig('num_utterances_log.png')

# see the players with top 10 number of utterances
count = 0
for name, num in num_utterances.items():
    print(name, num)
    count += 1
    if count > 10:
        break