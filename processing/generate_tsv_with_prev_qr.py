import os
import numpy as np
from tqdm import tqdm

os.chdir('../')

print('generating tsv file...')
with open(os.path.join('data', 'interview.txt'), 'r') as r:
    dataset = r.read()
    dataset = [interview for interview in dataset.split('[SEP]') if len(interview) > 10]  # don't include last one (\n)

    with open(os.path.join('data', 'interview_qa_with_prev_qr.tsv'), 'w') as w:
        w.write('id\t')
        w.write('sport_type\t')
        w.write('game_wiki_id\t')
        w.write('section_wiki_id\t')
        w.write('title\t')
        w.write('date\t')
        w.write('participants\t')
        w.write('background\t')
        w.write('prev_respondent\t')
        w.write('prev_question\t')
        w.write('prev_response\t')
        w.write('respondent\t')
        w.write('question\t')
        w.write('response\n')

        player2index = dict()

        for interview in tqdm(dataset):
            lines = interview.split('\n')
            # some lines may be empty due to splitting, remove those
            lines = [line.replace('End of FastScripts', '').strip() for line in lines if len(line) > 3]

            interview_id = lines[0][lines[0].index('[id]') + len('[id] '):]
            sport_type = lines[1][lines[1].index('[sport_type]') + len('[sport_type] '):]

            game_wiki_id = lines[2][lines[2].index('[game_wiki]') + len('[game_wiki] '):]
            section_wiki_id = lines[3][lines[3].index('[section_wiki]') + len('[section_wiki] '):]

            title = lines[4][lines[4].index('[title]') + len('[title] '):]
            date = lines[5][lines[5].index('[date]') + len('[date] '):]
            participants = lines[6][lines[6].index('[participants]') + len('[participants] '):]

            background = ''
            question_list = []
            response_list = []
            respondent_list = []
            for i in range(7, len(lines)):
                if '[background]' in lines[i]:
                    background += lines[i][lines[i].index('[background]') + len('[background] '):] + ' '
                elif '[QA]' in lines[i]:
                    if len(lines[i].split('\t')) < 2:  # some questions are thank-yous and do not have a response
                        continue

                    question, response = lines[i].split('\t')[:2]  # only keep the first response for every question
                    question = question[question.index('Q:') + len('Q: '):]

                    respondent = response[:response.index(':')].lower().title()
                    respondent_with_sport_type = respondent + '_' + sport_type
                    if respondent_with_sport_type not in player2index.keys():
                        player2index[respondent_with_sport_type] = len(player2index)
                    response = response[response.index(':') + len(': '):]
                    if len(question.strip()) == 0 or len(response.strip()) == 0:
                        continue

                    question_list.append(question)
                    response_list.append(response)
                    respondent_list.append(respondent + ' | ' + str(player2index[respondent_with_sport_type]))

            for i in range(1, len(question_list)):  # skip the first question bc it does not have a previous question
                w.write(f'{interview_id}\t')
                w.write(f'{sport_type}\t')
                w.write(f'{game_wiki_id}\t')
                w.write(f'{section_wiki_id}\t')
                w.write(f'{title}\t')
                w.write(f'{date}\t')
                w.write(f'{participants}\t')
                w.write(f'{background}\t')
                w.write(f'{respondent_list[i-1]}\t')
                w.write(f'{question_list[i-1]}\t')
                w.write(f'{response_list[i-1]}\t')
                w.write(f'{respondent_list[i]}\t')
                w.write(f'{question_list[i]}\t')
                w.write(f'{response_list[i]}\n')

        print(f'There are {len(player2index)} unique players.')


print('generating train, dev and test splits...')
with open(os.path.join('data', 'interview_qa_with_prev_qr.tsv'), 'r') as r:
    header = r.readline()
    lines = r.read()
    lines = [line for line in lines.split('\n') if len(line) > 3]

    shuffle_indices = np.random.choice(len(lines), len(lines), replace=False)

    idx = 0
    for split, percentage in zip(['train', 'dev', 'test'], [0.98, 0.99, 1]):
        with open(os.path.join('data', f'interview_qa_{split}_with_prev_qr.tsv'), 'w') as w:
            w.write(f'{header}')
            while idx < percentage * len(lines):
                w.write(f'{lines[shuffle_indices[idx]]}\n')
                idx += 1
