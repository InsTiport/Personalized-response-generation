import os
import wikipediaapi
from urllib.parse import unquote
import json
import sys
import pandas

os.chdir('../')
os.makedirs(os.path.join('data', 'wiki'), exist_ok=True)

title2index = dict()
game_search_results = json.loads(open(os.path.join("scraping", "game_search_result")).readline())
section_search_results = json.loads(open(os.path.join("scraping", "section_search_result")).readline())
 
for title in game_search_results:
    title2index[title] = len(title2index)

for title in section_search_results:
    title2index[title] = len(title2index)

index2title = {value: key for key, value in title2index.items()}

downloaded = set()
for entry in os.scandir(os.path.join('data', 'wiki')):
    if os.path.isfile(entry):
        downloaded.add(entry.name)

if len(downloaded) < len(title2index):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    for wiki_index in range(len(downloaded), len(title2index)):
        print(wiki_index)
        if wiki_index < len(game_search_results):
            result = game_search_results[index2title[wiki_index]]
            wiki_page = wiki_wiki.page(result[0]['title'])
        else:
            result = section_search_results[index2title[wiki_index]][0]
            title = unquote(result.split('/')[-1]).replace('_', ' ')
            wiki_page = wiki_wiki.page(title)
        with open(os.path.join('data', 'wiki', str(wiki_index)), 'w') as wiki_file:
            print(wiki_page.text)
            wiki_file.writelines(wiki_page.text)

def get_wiki_page(title):
    if title not in title2index:
        return -1, []
    else:
        wiki_index = title2index[title]
        return wiki_index, open(os.path.join('data', 'wiki', str(wiki_index)), 'r').readlines()