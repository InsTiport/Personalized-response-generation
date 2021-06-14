import datetime
import os
import re
from datetime import date
import numpy as np
from tqdm import tqdm
import spacy

spacy.cli.download('en_core_web_sm')
nlp = spacy.load('en_core_web_sm')

month_lookup = {
    'january': '01',
    'february': '02',
    'march': '03',
    'april': '04',
    'may': '05',
    'june': '06',
    'july': '07',
    'august': '08',
    'september': '09',
    'october': '10',
    'november': '11',
    'december': '12'
}

rev_month_lookup = {
    1: 'january',
    2: 'february',
    3: 'march',
    4: 'april',
    5: 'may',
    6: 'june',
    7: 'july',
    8: 'august',
    9: 'september',
    10: 'october',
    11: 'november',
    12: 'december'
}


def get_matching_news(sport, y, m, d, interview_title):
    m = m.lower()
    event_day = d if len(d) > 1 else '0' + d

    event_date = date.fromisoformat(f'{y}-{month_lookup[m]}-{event_day}')
    news_titles = []

    for news_title in os.scandir(os.path.join('data', 'espn', sport, y, m, d)):
        news_titles.append((news_title.name, event_date))

    next_day = event_date + datetime.timedelta(days=1)
    for news_title in os.scandir(os.path.join(
            'data', 'espn', sport, str(next_day.year),
            str(rev_month_lookup[next_day.month]),
            str(next_day.day)
    )):
        news_titles.append((news_title.name, next_day))

    matching_res = []
    nps = [str(s).lower() for s in nlp(interview_title).ents]
    print(nps)
    for news_title, event_date in news_titles:
        news_title_nps = [str(s).lower() for s in nlp(news_title).ents]
        for i, s in enumerate(news_title_nps):
            if '-' in s:
                news_title_nps[i] = s[s.index('-') + 2:]

        print(news_title, ' : ', news_title_nps)
        for np in nps:
            if np in news_title_nps:
                with open(os.path.join(
                        'data', 'espn', sport, str(event_date.year),
                        str(rev_month_lookup[event_date.month]),
                        str(event_date.day), news_title
                )) as f:
                    text = f.read()

                matching_res.append((news_title, text))

    return matching_res


def main():
    os.chdir('../')
    print('generating tsv file...')
    with open(os.path.join('data', 'interview.txt'), 'r') as r:
        dataset = r.read()
        dataset = [interview for interview in dataset.split('[SEP]') if len(interview) > 10]  # don't include last one (\n)

        with open(os.path.join('data', 'interview_qa_with_espn.tsv'), 'w') as w:
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
                if sport_type not in ['football', 'basketball', 'baseball', 'golf', 'hockey']:  # not linked to espn news
                    continue

                game_wiki_id = lines[2][lines[2].index('[game_wiki]') + len('[game_wiki] '):]
                section_wiki_id = lines[3][lines[3].index('[section_wiki]') + len('[section_wiki] '):]

                title = lines[4][lines[4].index('[title]') + len('[title] '):]
                date = lines[5][lines[5].index('[date]') + len('[date] '):]
                participants = lines[6][lines[6].index('[participants]') + len('[participants] '):]

                # link to espn
                month = re.search(r'[a-zA-Z]+', date).group(0)
                day_and_year = date[len(month):]
                day = day_and_year[:day_and_year.index(',')].strip()
                year = day_and_year[day_and_year.index(',') + 1:].strip()

                matched_news = get_matching_news(sport_type, year, month, day, title)

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

                for i in range(1, len(question_list)):  # skip the first question bc it's the first
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
    with open(os.path.join('data', 'interview_qa_with_espn.tsv'), 'r') as r:
        header = r.readline()
        lines = r.read()
        lines = [line for line in lines.split('\n') if len(line) > 3]

        shuffle_indices = np.random.choice(len(lines), len(lines), replace=False)

        idx = 0
        for split, percentage in zip(['train', 'dev', 'test'], [0.98, 0.99, 1]):
            with open(os.path.join('data', f'interview_qa_{split}_with_espn.tsv'), 'w') as w:
                w.write(f'{header}')
                while idx < percentage * len(lines):
                    w.write(f'{lines[shuffle_indices[idx]]}\n')
                    idx += 1

if __name__ == '__main__':
    main()