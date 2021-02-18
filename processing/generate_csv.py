import pandas as pd
import os
import re

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
    sports_type = ['football']

    # file system routine
    os.chdir('../')

    # initialize counter for interview episode
    episode_id = 0

    # store processed interviews here to avoid duplicates
    already_seen = set()

    '''
    For each sport type, process all interviews and put relevant information into dictionaries
    '''
    for sport in sports_type:
        sport_folder_path = os.path.join('data', sport)
        # loop through all player folders for this type of sport to include all interviews
        for player_folder in os.scandir(sport_folder_path):
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

                    # check if this interview has been processed before
                    if interview_text not in already_seen:
                        already_seen.add(interview_text)
                    else:
                        continue

                    # deal with names containing commas, e.g., A.J. Westbrook
                    interview_text = re.sub(r'([A-Z])\.([A-Z])\.', r'\1\2', interview_text)

                    '''
                    put relevant information into dictionaries
                    '''
                    episode['episode_id'].append(episode_id)
                    episode['is_interview'].append('conference' not in title.lower())
                    episode['title'].append(title)
                    episode['date'].append(date)
                    interviewees = process_text(interview_text, interviewees, episode_id)
                    episode['participants'].append('|'.join(interviewees))

                    # increase counter
                    episode_id += 1
                    # close file
                    f.close()
                    break
            break
        break

    df = pd.DataFrame(episode)
    df.to_csv(os.path.join('data', 'episode.csv'), index=False)
    df = pd.DataFrame(utterance)
    df.to_csv(os.path.join('data', 'utterance.csv'), index=False)


def process_text(text, speakers, episode_id):
    # counters
    speakers = [s.lower() for s in speakers]
    turn_id = -1
    turn_order = None
    current_speaker = None

    # do some 'fancy' processing
    text = text.replace('.', '..')
    text = text.replace('?', '??')
    text = text.replace(':', '::')
    text = text.replace('\n', '\n\n')
    text = text.replace('!', '!!')
    text = re.sub(r'(\.)\.', r'\1@', text)
    text = re.sub(r'(\?)\?', r'\1@', text)
    text = re.sub(r'(:):', r'\1@', text)
    text = re.sub(r'(\n)\n', r'\1@', text)
    text = re.sub(r'(!)!', r'\1@', text)
    # split the raw text based on several delimiters
    after_split = text.split('@')
    after_split = [s.strip() for s in after_split if len(s.strip()) > 1]

    # find utterances one by one, also keep track of turns
    for i, token in enumerate(after_split):
        if i == 0 and not token.isupper():
            continue

        # check if a token represents a change of turn or is an utterance
        if token.isupper():
            # often, there is one ':' or '.' after a person's name
            token = token[:-1]

            '''
            deals with various kinds of names a person may have
            '''
            # most common case
            if token.lower() in speakers:
                current_speaker = token.lower().title()
            # in case of interviewer, denote him/her by a special token
            elif token == 'Q' or token == 'THE MODERATOR':
                current_speaker = '[HOST]'
            # use partial match to pair some names, e.g., Coach Adams and John Adams
            elif partial_match(token, speakers) >= 0:
                current_speaker = speakers[partial_match(token, speakers)].title()
                # print(current_speaker)
                # print(token.lower())
                # print('\n')
            # there are cases when an interviewee is not present in the speakers list
            elif len(token) > 1 and 'COACH' not in token:
                speakers.append(token.lower())
                current_speaker = token.lower().title()
            else:
                print(episode_id)
                print(episode['title'][-1])
                print(token)
                print(speakers)
                exit(0)

            # update counter
            turn_order = 0
            turn_id += 1
        else:
            if turn_order is None:
                print(episode_id)
                print(episode['title'][-1])
                print(speakers)
                print(after_split)
                print(text)
                exit(0)
            # put utterance into dictionary
            utterance['episode_id'].append(episode_id)
            utterance['turn_id'].append(turn_id)
            utterance['turn_order'].append(turn_order)
            utterance['speaker'].append(current_speaker)
            utterance['utterance'].append(token)
            turn_order += 1

    # the speakers list may have been updated, due to missing interviewee names
    return [s.title() for s in speakers]


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
