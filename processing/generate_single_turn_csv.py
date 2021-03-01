import os
import re
import sys
import csv
import tqdm
import pysbd
from langdetect import detect
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP

# dictionary to store interview utterances
utterance = {
    'episode_id': [],
    'turn_id': [],
    'interviewee': [],
    'question': [],
    'response': []
}


def main():
    # sentence boundary detection
    seg = pysbd.Segmenter(language="en", clean=False)

    # file system routine
    os.chdir('../')

    # dynamically find all sport categories available
    sports_type = list(ID_LOOKUP.keys())
    sports_type = [dir_name for dir_name in sports_type if os.path.exists(os.path.join('data', dir_name))]

    # keep track of number of utterances
    total_utterances = 0

    # initialize counter for interview episode
    episode_id = 0

    # store non-English interviews found
    not_en = set()

    # csv writer object
    fw = open(os.path.join('data', 'csv', 'single_turn_utterance.csv'), 'w')
    csv_writer = csv.writer(fw)
    # write header to file
    csv_writer.writerow(['episode_id', 'turn_id', 'question', 'response'])

    '''
    For each sport type, process all interviews and put relevant information into dictionaries
    '''
    for sport in sports_type:
        # store processed interviews here to avoid duplicates
        already_seen = set()

        print(f'Generating csv files for {sport}...')
        sport_folder_path = os.path.join('data', sport)

        # loop through all player folders for this type of sport to include all interviews
        for player_folder in tqdm.tqdm(os.scandir(sport_folder_path)):
            # dealing with .DS_Store, make sure it's a folder
            if not os.path.isfile(player_folder):
                # process a single interview/conference each iteration
                for interview in os.scandir(player_folder):
                    fo = open(interview)

                    '''
                    get relevant information
                    '''
                    title = fo.readline()[:-1]
                    date = fo.readline()[:-1]
                    # remove repeating \n's and also replace the French 'e' with normal 'e'
                    remaining_text = re.sub(r'\n+', '\n', fo.read()).strip().replace('Ã©', 'e')
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
                    turn_count = generate_utterance(csv_writer, sentences, interviewees, episode_id)

                    # increase counter and close file
                    episode_id += 1
                    fo.close()
                    total_utterances += turn_count

    # close csv writer
    fw.close()

    # write non-English interviews discovered to file
    with open(os.path.join('data', 'csv', 'non_English_interviews.txt'), 'w') as f:
        for interview in not_en:
            f.write(interview.path + '\n')
        f.close()

    print(f'Processed {episode_id} interviews in total.')
    print(f'Generated {total_utterances} utterances in total.')


def generate_utterance(csv_writer, sentences, speakers, episode_id):
    # counters
    speakers = [s.lower() for s in speakers]
    turn_id = -1
    current_speaker = None
    previous_speaker = None

    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]

    # one question-response pair
    qa_pair = {
        'q': '',
        'a': ''
    }

    # find utterances one by one, also keep track of turns
    for i, sentence in enumerate(sentences):
        start_of_turn = check_start_of_turn(sentence)
        if turn_id < 0 and not start_of_turn:
            continue

        sentence = sentence.strip()
        # check if a token represents a change of turn or is an utterance
        if start_of_turn:
            speaker, sentence = start_of_turn
            sentence = sentence.strip()
            temp = current_speaker
            '''
            deals with various kinds of names a person may have
            '''
            # most common case
            if speaker.lower() in speakers:
                current_speaker = speaker.lower().title()
                if qa_pair['q'] == '':  # there is no question, so just skip
                    continue
                else:  # indicates start of response
                    qa_pair['a'] += sentence

            elif speaker == 'A':
                current_speaker = previous_speaker
                qa_pair['a'] += sentence  # since this is an 'A', this must be a response

            # in case of interviewer, denote him/her by a special token
            elif speaker == 'Q':
                current_speaker = '[Q]'

                if qa_pair['a'] != '':  # since there is a response, this must be a new turn
                    # write previous qa pair to file
                    csv_writer.writerow([episode_id, turn_id, qa_pair['q'], qa_pair['a']])
                    # clear storage
                    qa_pair = qa_pair.fromkeys(qa_pair, '')
                    # update q
                    qa_pair['q'] += sentence
                    # update turn
                    turn_id += 1
                else:  # this should be the very beginning of an interview
                    # update turn
                    turn_id += 1
                    # update q
                    qa_pair['q'] += sentence

            # in case of moderator, denote him/her by a special token
            elif speaker == 'THE MODERATOR':
                if qa_pair['q'] != '' and qa_pair['a'] != '':
                    print('here')
                    csv_writer.writerow([episode_id, turn_id, qa_pair['q'], qa_pair['a']])
                # clear storage
                qa_pair = qa_pair.fromkeys(qa_pair, '')

            # use partial match to pair some names, e.g., Coach Adams and John Adams
            elif partial_match(speaker, speakers) >= 0:
                current_speaker = speakers[partial_match(speaker, speakers)].title()
                if qa_pair['q'] == '':  # there is no question, so just skip
                    continue
                else:  # indicates start of response
                    qa_pair['a'] += sentence

            # there are cases when an interviewee is not present in the speakers list
            elif len(speaker) > 1:
                speakers.append(speaker.lower())
                current_speaker = speaker.lower().title()
                if qa_pair['q'] == '':  # there is no question, so just skip
                    continue
                else:  # indicates start of response
                    qa_pair['a'] += sentence

            # maybe there is a typo, just skip
            else:
                continue

            # update previous speaker
            previous_speaker = temp

        else:  # indicates that this sentence is part of someone's speaking
            if qa_pair['a'] != '':  # indicates that the interviewee is speaking
                qa_pair['a'] += ' ' + sentence
            else:  # indicates that the interviewer is speaking
                qa_pair['q'] += ' ' + sentence

    # write last qa pair to file
    if qa_pair['q'] != '' and qa_pair['a'] != '':
        csv_writer.writerow([episode_id, turn_id, qa_pair['q'], qa_pair['a']])

    return turn_id


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
