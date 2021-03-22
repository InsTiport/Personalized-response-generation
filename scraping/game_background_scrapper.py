import os
import json
import wikipediaapi
import requests
import sys

sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP


# Specifications. Change based on possible sport names (see scraper.py)
####################################################################
sports_type = list(ID_LOOKUP.keys())
####################################################################

# file system routine
os.chdir('../')

interview_count = 0
player_count = 0
games_list = set()
player_names = set()
for sport in sports_type:
    SPORT_FOLDER_PATH = os.path.join('data', sport)
    for player_folder in os.scandir(SPORT_FOLDER_PATH):
        player_names.add(player_folder.name + '_' + sport)
        player_count += 1
        if os.path.isdir(player_folder):
            for interview_text in os.scandir(player_folder):
                if interview_text.name.isnumeric():
                    interview_count += 1
                    with open(interview_text) as f:
                        game_title = f.readline().strip()
                        games_list.add(game_title)

game_types_list = set()
for game in games_list:
    game_types_list.add(game.split(':')[0])

print(f"{player_count} players")
print(f"There are {len(games_list)} distinct games among {interview_count} interviews.")
print(f"There are {len(game_types_list)} distinct game types in total")

if not os.path.exists(os.path.join('scraping', 'player_search_result')):
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": "", # query
        "srlimit": 1 # max number of pages to return
    }

    count = 0
    wiki_wiki = wikipediaapi.Wikipedia('en')
    search_result = dict()
    for name in player_names:
        PARAMS['srsearch'] = name
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        result_list = DATA['query']['search']
        if len(result_list) == 0:
            print("no match for", name)
        else:
            wiki_page = wiki_wiki.page(result_list[0]['title'])
            if name.split('_')[1] in wiki_page.summary and name.split('_')[0].split(',')[0] in wiki_page.summary:
                search_result[name] = result_list
                count += 1
                print(name + "       ====       " +  result_list[0]['title'])
    print(f'{count} out of {player_count} players found')
    file_output = open(os.path.join('scraping', 'player_search_result'), 'x')
    json.dump(search_result, file_output)

if not os.path.exists(os.path.join("scraping", "game_search_result")):
    # search pages on Wikipedia
    S = requests.Session()
    URL = "https://en.wikipedia.org/w/api.php"
    PARAMS = {
        "action": "query",
        "format": "json",
        "list": "search",
        "srsearch": "", # query
        "srlimit": 10 # max number of pages to return
    }

    search_result = dict()
    for game_type in game_types_list:
        PARAMS["srsearch"] = game_type
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        result_list = DATA['query']['search']
        if len(result_list) == 0:
            print("no match for", game_type)
        else:
            search_result[game_type] = result_list
            # print(game_type + "       :::::       " +  result_list[0]['title'])
    file_output = open(os.path.join("scraping", "game_search_result"), "x")
    json.dump(search_result, file_output)
else:
    search_result = json.loads(open(os.path.join("scraping", "game_search_result")).readline())

for sport in sports_type:
    SPORT_FOLDER_PATH = os.path.join('data', sport)
    for player_folder in os.scandir(SPORT_FOLDER_PATH):
        if os.path.isdir(player_folder):
            for file_ in os.scandir(player_folder):
                if file_.name.isnumeric():
                    context_file = open(os.path.join(os.path.dirname(file_), 'wiki_result_' + file_.name), 'w')
                    file_in = open(file_)
                    game_type = file_in.readline().strip().split(':')[0]
                    if game_type in search_result:
                        json.dump(search_result[game_type], context_file)
                    context_file.close()
                    file_in.close()
                    
print(f"There are {len(search_result)} game types that have linked wikipedia pages.")