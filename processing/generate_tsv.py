import os

os.chdir('../')

with open(os.path.join('data', 'interview.txt'), 'r') as r:
    dataset = r.read()
    dataset = [interview for interview in dataset.split('[SEP]') if len(interview) > 10]  # don't include last one (\n)

    with open(os.path.join('data', 'csv', 'interview_qa.tsv'), 'w') as w:
        w.write('id\t')
        w.write('sport_type\t')
        w.write('game_wiki\t')
        w.write('section_wiki\t')
        w.write('title\t')
        w.write('date\t')
        w.write('participants\t')
        w.write('background\t')
        w.write('respondent\t')
        w.write('question\t')
        w.write('response\n')

        for interview in dataset:
            lines = interview.split('\n')[:-1]  # remove the last line which contains only \n
            lines = [line.replace('End of FastScripts', '').strip() for line in lines if len(line) > 3]  # remove empty lines

            interview_id = lines[0][lines[0].index('[id]') + len('[id] '):]
            sport_type = lines[1][lines[1].index('[sport_type]') + len('[sport_type] '):]
            game_wiki = lines[2][lines[2].index('[game_wiki]') + len('[game_wiki] '):]
            section_wiki = lines[3][lines[3].index('[section_wiki]') + len('[section_wiki] '):]
            title = lines[4][lines[4].index('[title]') + len('[title] '):]
            date = lines[5][lines[5].index('[date]') + len('[date] '):]
            participants = lines[6][lines[6].index('[participants]') + len('[participants] '):]

            background = ''
            for i in range(7, len(lines)):
                if '[background]' in lines[i]:
                    background += lines[i][lines[i].index('[background]') + len('[background] '):] + ' '
                elif '[QA]' in lines[i]:
                    if len(lines[i].split('\t')) < 2:  # some questions are thank-yous and do not have a response
                        continue

                    question, response = lines[i].split('\t')[:2]  # only keep the first response for every question
                    question = question[question.index('Q:') + len('Q: '):]
                    respondent = response[:response.index(':')].lower().title()
                    response = response[response.index(':') + len(': '):]

                    w.write(f'{interview_id}\t')
                    w.write(f'{sport_type}\t')
                    w.write(f'{game_wiki}\t')
                    w.write(f'{section_wiki}\t')
                    w.write(f'{title}\t')
                    w.write(f'{date}\t')
                    w.write(f'{participants}\t')
                    w.write(f'{background}\t')

                    w.write(f'{respondent}\t')
                    w.write(f'{question}\t')
                    w.write(f'{response}\n')
