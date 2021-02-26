import pandas as pd
import os
import re
import sys
import tqdm
import pysbd
from langdetect import detect
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

# dictionary to store interview utterances
utterance = {
    'episode_id': [],
    'turn_id': [],
    'turn_order': [],
    'speaker': [],
    'utterance': []
}

# dictionary to store interview episodes information
episode = {
    'episode_id': [],
    'is_interview': [],
    'title': [],
    'date': [],
    'participants': []
}


def main():
    # sentence boundary detection
    seg = pysbd.Segmenter(language="en", clean=False)

    # file system routine
    os.chdir('../')

    # dynamically find all sport categories available
    sports_type = list(ID_LOOKUP.keys())
    sports_type = [dir_name for dir_name in sports_type if os.path.exists(os.path.join('data', dir_name))]

    # keep track of number of episodes and utterances
    total_episodes = 0
    total_utterances = 0

    # initialize counter for interview episode
    episode_id = 0

    '''
    For each sport type, process all interviews and put relevant information into dictionaries
    '''
    for sport in sports_type:
        # store processed interviews here to avoid duplicates
        already_seen = set()
        # store non-English interviews found
        not_en = set()

        print(f'Generating csv files for {sport}...')
        sport_folder_path = os.path.join('data', sport)

        # loop through all player folders for this type of sport to include all interviews
        for player_folder in tqdm.tqdm(os.scandir(sport_folder_path)):
            # dealing with .DS_Store, make sure it's a folder
            if not os.path.isfile(player_folder):
                # process a single interview/conference each iteration
                for interview in os.scandir(player_folder):
                    f = open(interview)

                    '''
                    get relevant information
                    '''
                    title = f.readline()[:-1]
                    date = f.readline()[:-1]
                    # remove repeating \n's and also replace the French 'e' with normal 'e'
                    remaining_text = re.sub(r'\n+', '\n', f.read()).strip().replace('Ã©', 'e')
                    interviewees = remaining_text[:remaining_text.index('START_OF_INTERVIEW_TEXT') - 1]
                    interview_text = remaining_text[len(interviewees) + len('START_OF_INTERVIEW_TEXT') + 2:
                                                    remaining_text.index('END_OF_INTERVIEW_TEXT')]
                    interviewees = interviewees.split('\n')

                    # check if this interview is in English
                    if detect(interview_text) != 'en':
                        not_en.add(interview)
                        continue

                    # check whether this interview has been encountered before
                    if interview_text[:100] not in already_seen:
                        already_seen.add(interview_text[:100])
                    else:
                        continue

                    try:
                        # break the interview text into sentences
                        sentences = list(seg.segment(interview_text.replace('\x1e', ' ')))
                    except ValueError:
                        print('Exception occurs')
                        print(title)
                        print(interviewees)

                    '''
                    put relevant information into dictionaries
                    '''
                    episode['episode_id'].append(episode_id)
                    episode['is_interview'].append('conference' not in title.lower())
                    episode['title'].append(title)
                    episode['date'].append(date)
                    interviewees = generate_utterance(sentences, interviewees, episode_id)
                    episode['participants'].append('|'.join(interviewees))

                    # increase counter and close file
                    episode_id += 1
                    f.close()

        # write to files
        os.makedirs(os.path.join('data', 'csv') + '/', exist_ok=True)
        df = pd.DataFrame(episode)
        df.to_csv(os.path.join('data', 'csv', f'{sport}_episode.csv'), index=False)
        df = pd.DataFrame(utterance)
        df.to_csv(os.path.join('data', 'csv', f'{sport}_utterance.csv'), index=False)
        # write non-English interviews discovered to file
        with open(os.path.join(sport_folder_path, 'non_English_interviews.txt'), 'w') as f:
            for interview in not_en:
                f.write(interview.path + '\n')
            f.close()

        # update counters
        total_episodes += len(episode['episode_id'])
        total_utterances += len(utterance['utterance'])
        # reset dictionaries and sets
        for e in episode:
            episode[e] = []
        for u in utterance:
            utterance[u] = []

    print(f'Processed {total_episodes} interviews in total.')
    print(f'Generated {total_utterances} utterances in total.')


def generate_utterance(sentences, speakers, episode_id):
    # counters
    speakers = [s.lower() for s in speakers]
    turn_id = -1
    turn_order = None
    current_speaker = None
    previous_speaker = None

    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]

    # find utterances one by one, also keep track of turns
    for i, sentence in enumerate(sentences):
        start_of_turn = check_start_of_turn(sentence)
        if turn_id < 0 and not start_of_turn:
            continue

        sentence = sentence.strip()
        # check if a token represents a change of turn or is an utterance
        if start_of_turn:
            speaker, sentence = start_of_turn
            temp = current_speaker
            '''
            deals with various kinds of names a person may have
            '''
            # most common case
            if speaker.lower() in speakers:
                current_speaker = speaker.lower().title()
            elif speaker == 'A':
                current_speaker = previous_speaker
            # in case of interviewer, denote him/her by a special token
            elif speaker == 'Q':
                current_speaker = '[Q]'
            # in case of moderator, denote him/her by a special token
            elif speaker == 'THE MODERATOR':
                current_speaker = '[MODERATOR]'
            # use partial match to pair some names, e.g., Coach Adams and John Adams
            elif partial_match(speaker, speakers) >= 0:
                current_speaker = speakers[partial_match(speaker, speakers)].title()
            # there are cases when an interviewee is not present in the speakers list
            elif len(speaker) > 1:
                speakers.append(speaker.lower())
                current_speaker = speaker.lower().title()
            # maybe there is a typo, just skip
            else:
                continue

            sentence = sentence.strip()

            # update counter and previous speaker
            turn_order = 0
            turn_id += 1
            previous_speaker = temp

        # skip texts before any one speaks, possibly skipping game results
        if turn_order is None:
            continue
        # put utterance into dictionary
        utterance['episode_id'].append(episode_id)
        utterance['turn_id'].append(turn_id)
        utterance['turn_order'].append(turn_order)
        utterance['speaker'].append(current_speaker)
        utterance['utterance'].append(sentence.replace('(', '').replace(')', ''))
        turn_order += 1

    # the speakers list may have been updated, due to missing interviewee names
    return [s.title() for s in speakers]


def check_start_of_turn(s):
    s = s.strip()

    # Q. or A.
    if s[:2] == 'Q.' or s[:2] == 'A.':
        return [s[0], s[2:]]

    # other cases have a semicolon present
    if ':' not in s:
        return False
    semicolon_idx = s.index(':')
    if not s[:semicolon_idx].isupper():
        return False
    return [s[:semicolon_idx], s[semicolon_idx + 1:]]


def partial_match(name_to_match, names):
    name_to_match = name_to_match.lower()
    names = [name.lower() for name in names]

    for index, name in enumerate(names):
        for part in name_to_match.split():
            if part in name:
                return index

    return -1


if __name__ == '__main__':
    main()
