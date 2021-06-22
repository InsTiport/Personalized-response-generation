# All corpus analytics we need in one script
import sys
import os
import json
import pandas as pd
from collections import OrderedDict
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

# file system routine
os.chdir('../')

interviewees = set() # names of all interviewees
interviewee_interview_count = dict() # interviewee --> count

for mode in ['train', 'test', 'dev']:
    with open(os.path.join('data', f'interview_qa_{mode}_with_prev_qr.tsv')) as f:
        f.readline()
        for line in f:
            interviewee_name = line.split('\t')[-3]
            interviewees.add(interviewee_name)
            if interviewee_name in interviewee_interview_count:
                interviewee_interview_count[interviewee_name].add(line.split('\t')[0])
            else:
                interviewee_interview_count[interviewee_name] = set()
interview_count = [len(s) for name, s in interviewee_interview_count.items()]
plt.hist(interview_count, bins=30, facecolor='blue', edgecolor='black', histtype='bar', alpha=0.5, cumulative=False, density=False)
plt.xlabel('Number of interview session attended')
plt.ylabel('Number of interviewees')
plt.yscale('log')
plt.savefig(os.path.join('figures', 'interview_count'))

# interview_num_stat = dict() # interview_count --> num_interviewees
# for interviewee, count in interviewee_interview_count.items():
#     if count not in interview_num_stat:
#         interview_num_stat[count] = 1
#     else:
#         interview_num_stat[count] += 1
# interview_num_stat = OrderedDict(sorted(interview_num_stat.items(), key=lambda t: t[0]))

# count_list = list()
# for interviewee, count in interviewee_interview_count.items():
#     count_list.append(count)

interviewee_utterance_count = dict()
for mode in ['train', 'test', 'dev']:
    with open(os.path.join('data', f'interview_qa_{mode}_with_prev_qr.tsv')) as f:
        f.readline()
        for line in f:
            interviewee_name = line.split('\t')[-3]
            if interviewee_name in interviewee_utterance_count:
                interviewee_utterance_count[interviewee_name] += 1
            else:
                interviewee_utterance_count[interviewee_name] = 1
print(len(interviewees), len(interviewee_utterance_count))

utterance_count = [count for name, count in interviewee_utterance_count.items() if count <= 5000]
print(len(utterance_count) / len(interviewee_utterance_count))
plt.figure()
values, bins, _ = plt.hist(utterance_count, bins=30, facecolor='blue', edgecolor='black', histtype='bar', alpha=0.5, cumulative=False, density=False)
plt.xlabel('Number of utterances spoken')
plt.ylabel('Number of interviewees')
plt.yscale('log')
plt.savefig(os.path.join('figures', 'utterance_count'))