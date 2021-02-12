import os
import wikipediaapi
import requests

# Specifications. Change based on possible sport names (see scraper.py)
####################################################################
sport_type = 'football'
####################################################################

SPORT_FOLDER_PATH = os.path.join('data', sport_type)

interview_count = 0
games_list = set()
for player_folder in os.scandir(SPORT_FOLDER_PATH):
    for interview_text in os.scandir(player_folder):
        interview_count += 1
        with open(interview_text) as f:
            game_title = f.readline().strip()
            games_list.add(game_title)

game_types_list = set()
for game in games_list:
    game_types_list.add(game.split(':')[0])


print(f"There are {len(games_list)} distinct {sport_type} games among {interview_count} interviews.")
print(f"There are {len(game_types_list)} distinct {sport_type} game types in total")


# search pages on Wikipedia
S = requests.Session()
URL = "https://en.wikipedia.org/w/api.php"
PARAMS = {
    "action": "opensearch",
    "namespace": "0",
    "search": "",
    "format": "json",
}

pages_found_count = 0
for game_type in game_types_list:
    PARAMS["search"] = game_type
    R = S.get(url=URL, params=PARAMS)
    DATA = R.json()
    if len(DATA[1]) != 0:
        pages_found_count += 1 
        print(DATA)

print(f"There are {pages_found_count} game types that have linked wikipedia pages.")