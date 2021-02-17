import os
import json
import wikipediaapi
import requests

# Specifications. Change based on possible sport names (see scraper.py)
####################################################################
sport_type = 'football'
####################################################################

SPORT_FOLDER_PATH = os.path.join('../data', sport_type)

interview_count = 0
player_count = 0
games_list = set()
for player_folder in os.scandir(SPORT_FOLDER_PATH):
    player_count += 1
    for interview_text in os.scandir(player_folder):
        interview_count += 1
        with open(interview_text) as f:
            game_title = f.readline().strip()
            games_list.add(game_title)

game_types_list = set()
for game in games_list:
    game_types_list.add(game.split(':')[0])

print(f"{player_count} players")
print(f"There are {len(games_list)} distinct {sport_type} games among {interview_count} interviews.")
print(f"There are {len(game_types_list)} distinct {sport_type} game types in total")

if not os.path.exists(os.path.join("game_search_result")):
    # search pages on Wikipedia
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "opensearch",
        "namespace": "0",
        "search": "",
        "format": "json",
    }

    search_result = list()
    for game_type in game_types_list:
        PARAMS["search"] = game_type
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        if len(DATA[1]) != 0:
            search_result.append(DATA)
            print(DATA[0])
    file_output = open(os.path.join("game_search_result"), "x")
    json.dump(search_result, file_output)
else:
    search_result = json.loads(open(os.path.join("game_search_result")).readline())


print(f"There are {len(search_result)} game types that have linked wikipedia pages.")

wiki_wiki = wikipediaapi.Wikipedia('en')
os.makedirs(os.path.join('../data', 'wikipedia'), exist_ok=True)
for result in search_result:
    os.makedirs(os.path.join('../data', 'wikipedia', result[0]), exist_ok=True)
    game_path_name = os.path.join('../data', 'wikipedia', result[0])
    for page_name in result[1]:
        if not os.path.exists(os.path.join(game_path_name, page_name.replace('/', ' '))):
            open(os.path.join(game_path_name, page_name.replace('/', ' ')), 'x')
        wiki_page = wiki_wiki.page(page_name)
        with open(os.path.join(game_path_name, page_name.replace('/', ' ')), 'w') as f:
            f.write(wiki_page.summary)
        