import os
import re
import sys
import tqdm
import pysbd
from langdetect import detect
sys.path.insert(0, os.path.abspath('..'))
from scraping.scraper import ID_LOOKUP
from processing.utils import check_start_of_turn, partial_match
from processing.background_utils import get_wiki_index


def main():
    # sentence boundary detection
    seg = pysbd.Segmenter(language="en", clean=False)

    # dynamically find all sport categories available
    sports_type = list(ID_LOOKUP.keys())
    sports_type = [dir_name for dir_name in sports_type if os.path.exists(os.path.join('data', dir_name))]

    # keep track of the total number of question-answer pairs
    total_qa_pairs = 0

    # initialize counter for interview episode
    episode_id = 1

    # writer object
    dataset_writer = open(os.path.join('data', 'interview.txt'), 'w')

    '''
    For each sport type, process all interviews and put relevant information into dictionaries
    '''
    for sport in sports_type:
        # store processed interviews here to avoid duplicates
        already_seen = set()
        # store non-English interviews found
        not_en = set()

        print(f'Generating interview data for {sport}...')
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
                    interviewees = interviewees.split('\n')
                    interview_text = remaining_text[remaining_text.index('START_OF_INTERVIEW_TEXT') +
                                                    len('START_OF_INTERVIEW_TEXT'):
                                                    remaining_text.index('END_OF_INTERVIEW_TEXT')]
                    interview_text = interview_text.replace('\x1e', ' ')  # annoying character

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
                        sentences = list(seg.segment(interview_text))  # \x1e: record separator
                    except ValueError:
                        print('Exception occurs while doing sentence segmentation')
                        print(title)
                        print(interviewees)

                    '''
                    write relevant information into interview.txt
                    '''
                    dataset_writer.write(f'1 [id] {episode_id}\n')
                    dataset_writer.write(f'2 [sport_type] {sport}\n')
                    game_wiki = get_wiki_index(title[:title.index(':')]) if ':' in title else get_wiki_index(title)
                    dataset_writer.write(f'3 [game_wiki]{(" " + str(game_wiki)) if game_wiki != -1 else ""}\n')
                    section_wiki = get_wiki_index(title) if ':' in title else -1
                    dataset_writer.write(f'4 [section_wiki]{(" " + str(section_wiki)) if section_wiki != -1 else ""}\n')
                    dataset_writer.write(f'5 [title] {title}\n')
                    dataset_writer.write(f'6 [date] {date}\n')
                    interviewees, backgrounds, qa_pairs = generate_utterance(sentences, interviewees, episode_id)
                    dataset_writer.write(f'7 [participants] {"|".join(interviewees)}\n')
                    idx = 8
                    for bg in backgrounds:
                        dataset_writer.write(f'{idx} [background] {bg}\n')
                        idx += 1
                    for qa in qa_pairs:
                        question, answers = qa[0], '\t'.join(qa[1:])  # for multiple responses, join them with tabs
                        dataset_writer.write(f'{idx} [QA] {question}\t{answers}\n')
                        idx += 1
                    dataset_writer.write('[SEP]\n')  # separator token between different interviews

                    # update counter
                    total_qa_pairs += len(qa_pairs)
                    episode_id += 1

                    f.close()

        # write non-English interviews discovered to file
        with open(os.path.join(sport_folder_path, 'non_English_interviews.txt'), 'w') as f:
            for interview in not_en:
                f.write(interview.path + '\n')
            f.close()

    print(f'Processed {episode_id} interviews in total.')
    print(f'Generated {total_qa_pairs} question-answer pairs in total.')


def generate_utterance(sentences, speakers, episode_id):
    # some basic processing on input sentences
    sentences = [s.strip() for s in sentences if len(s.strip()) > 1]

    # flags for the background part
    is_in_qa_part = False
    is_part_of_speaking = False
    # flags for the QA part
    current_speaker = None
    previous_speaker = None

    # lists to store relevant information
    backgrounds = []
    qas = []

    # one question-response pair
    qa_pair = {
        'q': '',
        'a': []  # there may be multiple answers from different interviewees
    }
    # to store one background comment at a time (effective before the QA part starts)
    comment = ''

    # find utterances one by one, also keep track of turns
    for i, sentence in enumerate(sentences):
        start_of_turn = check_start_of_turn(sentence)

        '''
        process sentences they belongs to the background part, which comes before the QA part
        '''
        if not is_in_qa_part:
            if not start_of_turn:
                if is_part_of_speaking:
                    comment += ' ' + sentence  # continuation of someone's speaking
                else:
                    backgrounds.append(sentence)  # plain background, like match scores
                continue
            else:
                if start_of_turn[0] != 'Q':  # indicates this comment belongs to the background part, not the qa part
                    if len(comment) > 0:
                        backgrounds.append(comment)  # add previous comment to background if it exists

                    is_part_of_speaking = True  # since someone is speaking, set the flag to true
                    comment = sentence  # reset :comment: to store a new piece of comment
                    continue
                else:  # marks the start of QA part
                    is_in_qa_part = True
                    if len(comment) > 0:
                        backgrounds.append(comment)  # add the last comment to background if it exists

        '''
        process sentences they belongs to the QA part, which comes after the background part
        '''
        # check if a token represents a change of turn or is an utterance
        if start_of_turn:
            speaker, sentence = start_of_turn
            sentence = sentence.strip()
            temp = current_speaker
            '''
            deals with various kinds of names a person may have
            '''
            # most common case
            if speaker in speakers:
                current_speaker = speaker

                qa_pair['a'].append(speaker + ': ' + sentence)

            elif speaker == 'A':
                current_speaker = previous_speaker
                qa_pair['a'].append(speaker + ': ' + sentence)  # since this is an 'A', this must be a response

            # in case of interviewer, denote him/her by a special token
            elif speaker == 'Q':
                current_speaker = '[Q]'

                if len(qa_pair['a']) > 0:  # since there is a response, this must be a new turn
                    # write previous qa pair to storage
                    qas.append((qa_pair['q'], *qa_pair['a']))
                    # clear storage
                    qa_pair['q'], qa_pair['a'] = '', []

                # update q
                qa_pair['q'] = 'Q: ' + sentence

            # # in case of moderator, denote him/her by a special token
            # elif speaker == 'THE MODERATOR':
            #     if qa_pair['q'] != '' and qa_pair['a'] != '':
            #         qas.append((qa_pair['q'], qa_pair['a']))
            #     # clear storage
            #     qa_pair = qa_pair.fromkeys(qa_pair, '')

            # use partial match to pair some names, e.g., Coach Adams and John Adams
            elif partial_match(speaker, speakers) >= 0:
                current_speaker = speakers[partial_match(speaker, speakers)]

                qa_pair['a'].append(current_speaker + ': ' + sentence)

            # there are cases when an interviewee is not present in the speakers list
            elif len(speaker) > 1:
                speakers.append(speaker)
                current_speaker = speaker

                qa_pair['a'].append(speaker + ': ' + sentence)

            # maybe there is a typo, just skip
            else:
                print(episode_id)  # for debugging
                continue

            # update previous speaker
            previous_speaker = temp

        else:  # indicates that this sentence is part of someone's speaking
            if len(qa_pair['a']) > 0:  # indicates that the interviewee is speaking
                qa_pair['a'][-1] += ' ' + sentence
            else:  # indicates that the interviewer is speaking
                qa_pair['q'] += ' ' + sentence

    # write last qa pair to file if it exists
    if qa_pair['q'] != '' and qa_pair['a'] != '':
        qas.append((qa_pair['q'], *qa_pair['a']))

    # the speakers list may have been updated, due to missing interviewee names
    return speakers, backgrounds, qas


if __name__ == '__main__':
    main()
