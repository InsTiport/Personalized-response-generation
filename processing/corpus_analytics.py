# All corpus analytics we need in one script
import sys
import os
import json
from collections import OrderedDict
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

# file system routine
os.chdir('../')

interviewees = set() # names of all interviewees
interview_count = dict() # interviewee --> count

sports_type = list(ID_LOOKUP.keys())
for sport in sports_type:
    SPORT_FOLDER_PATH = os.path.join('data', sport)
    for player_folder in os.scandir(SPORT_FOLDER_PATH):
        if player_folder.is_dir():
            name_with_sport = player_folder.name + '_' + sport
            interviewees.add(name_with_sport)
            interview_count[name_with_sport] = len([file for file in os.scandir(os.path.join(SPORT_FOLDER_PATH, player_folder.name)) if file.is_file()])

interview_num_stat = dict() # interview_count --> num_interviewees
for interviewee, count in interview_count.items():
    if count not in interview_num_stat:
        interview_num_stat[count] = 1
    else:
        interview_num_stat[count] += 1
interview_num_stat = OrderedDict(sorted(interview_num_stat.items(), key=lambda t: t[0]))

count_list = list()
for interviewee, count in interview_count.items():
    count_list.append(count)
print(interview_num_stat)
# plt.hist(count_list, color='blue', edgecolor='black', bins=int(1000))
# plt.show()