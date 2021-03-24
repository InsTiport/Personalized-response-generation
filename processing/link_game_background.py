import os
import wikipediaapi
import json
import sys
import pandas

sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

sports_type = list(ID_LOOKUP.keys())

os.chdir('../')
os.makedirs(os.path.join('data', 'wiki'), exist_ok=True)

wiki_page_index = 0 # correspond to the name of files in data/wiki
wiki_wiki = wikipediaapi.Wikipedia('en')

# link the wikipedia pages of general game types to each episode
game_search_results = json.loads(open(os.path.join("scraping", "game_search_result")).readline())
game_title_seen = set()
for sport in sports_type:
    episode_table = pandas.read_csv(os.path.join('data', 'csv', sport + '_episode.csv'))
    game_background = list()
    
    for _, row in episode_table.iterrows():
        try:
            title = row['title']
            title = title.split(':')[0]
            if title in game_search_results:
                if title not in game_title_seen:
                    game_title_seen.add(title)
                    wiki_page_index += 1
                    result = game_search_results[title]
                    wiki_page = wiki_wiki.page(result[0]['title'])
                    with open(os.path.join('data', 'wiki', str(wiki_page_index)), 'w') as wiki_file:
                        wiki_file.writelines(wiki_page.text)
                        pass
                    game_background.append(wiki_page_index)
                else:
                    game_background.append(wiki_page_index)
            else:
                game_background.append(None)
        except Exception as e:
            print(e)

    print(sport + " done")
    # save
    episode_table['game_background'] = game_background
    episode_table.to_csv(os.path.join('data', 'csv', sport + '_episode.csv'), index=False)

# link section wikipedia pages to each episode
section_search_results = json.loads(open(os.path.join("scraping", "section_search_result")).readline())
section_title_seen = set()
for sport in sports_type:
    episode_table = pandas.read_csv(os.path.join('data', 'csv', sport + '_episode.csv'))
    section_background = list()

    for _, row in episode_table.iterrows():
        try:
            title = row['title']
            if title in section_search_results:
                if title not in section_title_seen:
                    section_title_seen.add(title)
                    wiki_page_index += 1
                    result = section_search_results[title]
                    wiki_page = wiki_wiki.page(result[0])
                    with open(os.path.join('data', 'wiki', str(wiki_page_index)), 'w') as wiki_file:
                        wiki_file.writelines(wiki_page.text)
                        pass
                    section_background.append(wiki_page_index)
                else:
                    section_background.append(wiki_page_index)
            else:
                section_background.append(None)
        except Exception as e:
            print(e)
    
    print(sport + ' done')
    # save
    episode_table['section_background'] = section_background
    episode_table.to_csv(os.path.join('data', 'csv', sport + '_episode.csv'), index=False)

