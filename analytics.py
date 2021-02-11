import os
import numpy as np

# Specifications. Change based on possible sport names (see scrapper.py)
####################################################################
sport_type = 'football'
####################################################################

SPORT_FOLDER_PATH = os.path.join('data', sport_type)

interview_counts = []
interview_counts_player = dict()
for player_folder in os.scandir(SPORT_FOLDER_PATH):
    if not os.path.isfile(player_folder):
        number_of_interviews = len([name for name in os.scandir(player_folder) if os.path.isfile(name)])
        if number_of_interviews > 0:
            interview_counts.append(number_of_interviews)
            interview_counts_player[player_folder.name] = number_of_interviews

interview_counts = np.array(interview_counts)

print(f'There are a total of {len(interview_counts)} football players.')
print(f'The average number of interviews per player is {np.mean(interview_counts)}')
print(f'The minimum number of interviews one player has is {np.min(interview_counts)}')
print(f'The maximum number of interviews one player has is {np.max(interview_counts)}')
print(f'The player with {np.max(interview_counts)} interviews is'
      f' {max(interview_counts_player, key=lambda x: interview_counts_player[x])}')

print(
    f'There are {np.sum(interview_counts == 1)} players with only one interview, '
    f'which accounts for {100 * np.sum(interview_counts == 1) / len(interview_counts)}% of all players')
print(
    f'There are {np.sum(interview_counts <= 2)} players with less or equal than or equal to two interviews, '
    f'which accounts for {100 * np.sum(interview_counts <= 2) / len(interview_counts)}% of all players')
print(
    f'There are {np.sum(interview_counts <= 3)} players with less or equal than or equal to three interviews, '
    f'which accounts for {100 * np.sum(interview_counts <= 3) / len(interview_counts)}% of all players')
print(
    f'There are {np.sum(interview_counts <= 4)} players with less or equal than or equal to four interviews, '
    f'which accounts for {100 * np.sum(interview_counts <= 4) / len(interview_counts)}% of all players')
print(
    f'There are {np.sum(interview_counts <= 5)} players with less or equal than or equal to five interviews, '
    f'which accounts for {100 * np.sum(interview_counts <= 5) / len(interview_counts)}% of all players')
