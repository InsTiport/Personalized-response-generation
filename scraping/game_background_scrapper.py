import os
import json
import wikipediaapi
import requests
import sys
import time
from googlesearch import search

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

# scrape wikipedia pages for general game types
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
    with open(os.path.join("scraping", "game_search_result"), "x") as file_output:
        json.dump(search_result, file_output)

count = 0
for game_section in games_list:
    if ':' in game_section:
        count += 1
print(count)
section_search_result = json.loads(open(os.path.join('scraping', 'section_search_result'), 'r').readline())
print(len(section_search_result))

# scrape wikipedia pages for game sections
# if not os.path.exists(os.path.join("scraping", "section_search_result")):
#     section_search_result = dict()
# else:
#     section_search_result = json.loads(open(os.path.join('scraping', 'section_search_result'), 'r').readline())
# for game_section in games_list:
#     if ':' in game_section:
#         if game_section not in section_search_result:
#             try:
#                 PARAMS = {
#                     'key': 'AIzaSyAdwkEI2n6oemWHZ3SlaUA37UAaLb5E1ME',
#                     'cx': '6008a5bef15315ee8',
#                     'q': game_section,
#                     'start': 1,
#                     'num': 10,
#                     'lr': 'lang_en'
#                 }
#                 R = requests.get('https://www.googleapis.com/customsearch/v1', params=PARAMS)
#                 print(R)
#                 content = json.loads(R.content)
#                 url_list = list()
#                 for i in range(len(content['items'])):
#                     url_list.append(content['items'][i]['link'])
#                 section_search_result[game_section] = url_list
#                 print(game_section + "    ====    " + url_list[0])
#                 time.sleep(2)
#             except Exception as e:
#                 print(e)
# with open(os.path.join("scraping", "section_search_result"), "w") as file_output:
#     json.dump(section_search_result, file_output)


# scrape player wikipedia
if not os.path.exists(os.path.join('scraping', 'player_search_result')):
    player_search_result = dict()
else:
    player_search_result = json.loads(open(os.path.join('scraping', 'player_search_result'), 'r').readline())

count = 0
wiki_wiki = wikipediaapi.Wikipedia('en')
S = requests.Session()
for name in player_names:
    try:
        URL = "https://en.wikipedia.org/w/api.php"
        PARAMS = {
            "action": "query",
            "format": "json",
            "list": "search",
            "srsearch": name, # query
            "srlimit": 1 # max number of pages to return
        }
        R = S.get(url=URL, params=PARAMS)
        DATA = R.json()
        result_list = DATA['query']['search']
        if len(result_list) == 0:
            print("no match for", name)
        else:
            wiki_page = wiki_wiki.page(result_list[0]['title'])
            if name.split('_')[1] in wiki_page.summary and name.split('_')[0].split(',')[0] in wiki_page.summary:
                player_search_result[name] = result_list
                count += 1
                try:
                    print(name + "       ====       " +  result_list[0]['title'])
                except Exception as e:
                    print(e)
    except Exception as e:
        print(e)
with open(os.path.join('scraping', 'player_search_result'), 'x') as file_output:
    json.dump(player_search_result, file_output)

print(f'{count} out of {player_count} players found')
