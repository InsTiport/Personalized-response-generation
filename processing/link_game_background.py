import os
import wikipediaapi
import json
import sys
import pandas

sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

os.chdir('../')

sports_type = list(ID_LOOKUP.keys())
search_results = json.loads(open(os.path.join("scraping", "game_search_result")).readline())
csv_path = os.path.join('data', 'csv')

wiki_page_index = 0
os.makedirs(os.path.join('data', 'wiki'), exist_ok=True)
wiki_wiki = wikipediaapi.Wikipedia('en')
title_seen = set()
for sport in sports_type:
    episode_table = pandas.read_csv(os.path.join('data', 'csv', sport + '_episode.csv'))
    
    # add a column for background
    for index, row in episode_table.iterrows():
        try:
            title = row['title']
            if title in search_results:
                if title not in title_seen:
                    wiki_page_index += 1
                    result = search_results[title]
                    wiki_page = wiki_wiki.page(result[0]['title'])
                    with open(os.path.join('data', 'wiki', str(wiki_page_index)), 'w') as wiki_file:
                        wiki_file.writelines(wiki_page.text)
                else:
                    row['background'] = wiki_page_index

                title_seen.add(title)
        except:
            pass

    print(sport + " done")    
    # save
    episode_table.to_csv(os.path.join('data', 'csv', sport + '_episode.csv'))  
            