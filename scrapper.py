import os
import tqdm
from interview_link_scrapper import get_player_interview_links_for_one_sport
from interview_text_scrapper import get_interview_text
from typing import Dict, List


DATA_PATH = 'data/'
SPORT_PATH = 'football/'
sport_url = 'http://www.asapsports.com/showcat.php?id=1&event=yes'

os.makedirs(os.path.dirname(DATA_PATH + SPORT_PATH), exist_ok=True)

# get all interviews for all football players
player_interview_links: Dict[str, List[str]] = get_player_interview_links_for_one_sport(sport_url)

# write interviews to text files
for player, player_interview_urls in tqdm.tqdm(player_interview_links.items()):
    for interview_url in player_interview_urls:
        name, time, place, players, text = get_interview_text(interview_url)
        filename = DATA_PATH + SPORT_PATH + player
        with open(filename, 'w') as f:
            f.write(name + '\n')
            f.write(time + '\n')
            f.write(place + '\n')
            f.writelines(players)
            f.write('\n')
            f.write('START_OF_INTERVIEW_TEXT' + '\n')
            f.write(text + '\n')
            f.write('END_OF_INTERVIEW_TEXT')
            f.close()
