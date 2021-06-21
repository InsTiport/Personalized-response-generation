import os
import wikipediaapi
import json


os.chdir('../')
os.makedirs(os.path.join('data', 'interviewee_wiki'), exist_ok=True)

player_search_result = json.loads(open(os.path.join("scraping", "player_search_result")).readline())
interviewees = player_search_result.keys()

downloaded = set()
for entry in os.scandir(os.path.join('data', 'interviewee_wiki')):
    if os.path.isfile(entry):
        downloaded.add(entry.name)
print(f'{len(downloaded)} wikis downloaded')

print('Downloading wiki...')
count = 0
if len(downloaded) < len(interviewees):
    wiki_wiki = wikipediaapi.Wikipedia('en')
    for interviewee in interviewees:
        if interviewee not in downloaded:
            print(interviewee)
            result = player_search_result[interviewee]
            wiki_page = wiki_wiki.page(result[0]['title'])
            with open(os.path.join('data', 'interviewee_wiki', interviewee), 'w') as wiki_file:
                wiki_file.writelines(wiki_page.text.replace('\u2013', ''))
                count += 1
        if count % 100 == 0 and count > 0:
            print(count)
print('Finished downloading wiki.')


def convert_name(interviewee_name):
    name, sport_type = interviewee_name.split('_')
    components = name.split()
    if len(components) == 1:
        return interviewee_name
    else:
        return components[1] + ', ' + components[0] + f'_{sport_type}'


def get_wiki_page(interviewee_name):
    interviewee_name = convert_name(interviewee_name)
    if interviewee_name not in interviewees:
        return None
    else:
        return os.path.join('data', 'interviewee_wiki', interviewee_name)
        # return open(os.path.join('data', 'interviewee_wiki', interviewee_name), 'r').readlines()

    # print(get_wiki_page('Bo Pelini_football'))
    # print(get_wiki_page('Tracy Claeys_football'))
