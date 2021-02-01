import os
import tqdm
from interview_link_scrapper import get_player_interview_links_for_one_sport
from interview_text_scrapper import get_interview_text
from typing import Dict, List

# Specifications. Change if needed
####################################################################
DATA_FOLDER_PATH = 'data/'
SPORT_FOLDER_PATH = 'football/'
sport_url = 'http://www.asapsports.com/showcat.php?id=1&event=yes'
####################################################################


os.makedirs(os.path.dirname(DATA_FOLDER_PATH + SPORT_FOLDER_PATH), exist_ok=True)

# get all interviews for all football players
player_interview_links: Dict[str, List[str]] = get_player_interview_links_for_one_sport(sport_url)

# write interviews to text files
for player, player_interview_urls in tqdm.tqdm(player_interview_links.items()):
    os.makedirs(os.path.dirname(DATA_FOLDER_PATH + SPORT_FOLDER_PATH + player + '/'), exist_ok=True)
    count = 1
    for interview_url in player_interview_urls:
        name, time, players, text = get_interview_text(interview_url)
        filename = DATA_FOLDER_PATH + SPORT_FOLDER_PATH + player + '/' + str(count)
        count += 1
        with open(filename, 'w') as f:
            f.write(name + '\n')
            f.write(time + '\n')
            for player_name in players:
                f.write(player_name + '\n')
            f.write('START_OF_INTERVIEW_TEXT' + '\n')
            f.write(text + '\n')
            f.write('END_OF_INTERVIEW_TEXT')
            f.close()
