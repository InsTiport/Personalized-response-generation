import os
import sys
import json
import wikipediaapi
from urllib.parse import unquote

sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP
sports_type = list(ID_LOOKUP.keys())

os.chdir('../')
games_list = set()
for sport in sports_type:
    SPORT_FOLDER_PATH = os.path.join('data', sport)
    for player_folder in os.scandir(SPORT_FOLDER_PATH):
        if os.path.isdir(player_folder):
            for interview_text in os.scandir(player_folder):
                if interview_text.name.isnumeric():
                    with open(interview_text) as f:
                        game_title = f.readline().strip()
                        games_list.add(game_title)

non_game = set()
for title in games_list:
    if ':' not in title:
        non_game.add(title)
print(f'{len(non_game)} out of {len(games_list)} are NOT from an after game interview')

# measure the quality of the wikipedia pages linked to each section

game_search_result = json.loads(open(os.path.join('scraping', 'game_search_result'), 'r').readline())

conference_count = 0
media_day_count = 0
open_count = 0
announcement_count = 0
championship_count = 0
for name in non_game:
    if 'CONFERENCE' in name:
        conference_count += 1
    elif 'DAY' in name:
        media_day_count += 1
    elif 'OPEN' in name:
        open_count += 1
    elif 'ANNOUNCEMENT' in name:
        announcement_count += 1
    elif 'CHAMPIONSHIP' in name:
        championship_count += 1
    else:
        print(name)
print(f'{conference_count} MEDIA CONFERENCES')
print(f'{media_day_count} MEDIA DAYS')
print(f'{open_count} OPENS')
print(f'{announcement_count} ANNOUNCEMENTS')
print(f'{championship_count} CHAMPIONSHIPS')



section_search_result = json.loads(open(os.path.join('scraping', 'section_search_result'), 'r').readline())
wiki_wiki = wikipediaapi.Wikipedia('en')

count = 0
qualified_result_count = 0
for section_title in section_search_result:
    game_type = section_title.split(":")[0].strip()
    teams = section_title.split(":")[1]
    team1 = ''
    team2 = ''
    if ' v ' in teams:
        team1 = teams.split('v')[0].strip()
        team2 = teams.split('v')[1].strip()
    elif ' V ' in teams:
        team1 = teams.split('V')[0].strip()
        team2 = teams.split('V')[1].strip()
    elif ' vs ' in teams:
        team1 = teams.split('vs')[0].strip()
        team2 = teams.split('vs')[1].strip()
    elif ' VS ' in teams:
        team1 = teams.split('VS')[0].strip()
        team2 = teams.split('VS')[1].strip()
    else:
        team1 = teams.strip()
    
    title = section_search_result[section_title][0].split('/')[-1]
    title = unquote(title).replace('_', ' ')
    for i in range(100):
        try:
            content = wiki_wiki.page(title).text
            break
        except Exception as e:
            print(e)
    if team1.lower() in content.lower() and team2.lower() in content.lower():
        qualified_result_count += 1
    count += 1
    # print(title)
    # print(team1)
    # print(team2)
    # print(content)
    print(f'{qualified_result_count} / {count} matched')

print(f'{qualified_result_count} out of {len(section_search_result)}')
