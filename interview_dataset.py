import torch
import os
import linecache
import time
import matplotlib.pyplot as plt
from datasets import tqdm
import seaborn as sns
from train.SBERT_filtering import find_top_k


def matching_score(gold, ref):
    counter = 0
    for i in range(1, len(gold)):
        if gold[:i] in ref:
            counter += 1

    return counter / len(gold)


class InterviewDataset(torch.utils.data.Dataset):
    # data can be 'train', 'dev' or 'test'
    def __init__(self, data='train', use_wiki=False):
        self.filename = os.path.join('data', f'interview_qa_{data}.tsv')
        self.use_wiki = use_wiki
        with open(self.filename, 'r') as r:
            lines = r.read()
            self.len = len([line for line in lines.split('\n') if len(line) > 3]) - 1
        del lines

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        line = linecache.getline(self.filename, item + 2).strip().split('\t')

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return linecache.getline(self.filename, 1).strip().split('\t')


class InterviewDatasetWithPrevQR(torch.utils.data.Dataset):
    # data can be 'train', 'dev' or 'test'
    def __init__(self, data='train', use_wiki=False):
        self.filename = os.path.join('data', f'interview_qa_{data}_with_prev_qr.tsv')
        self.use_wiki = use_wiki
        with open(self.filename, 'r') as r:
            lines = r.read()
            self.len = len([line for line in lines.split('\n') if len(line) > 3]) - 1
        del lines

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        line = linecache.getline(self.filename, item + 2).strip().split('\t')

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return linecache.getline(self.filename, 1).strip().split('\t')


class InterviewDatasetESPN(torch.utils.data.Dataset):
    # data can be 'train', 'dev' or 'test'
    def __init__(self, data='train', use_wiki=False):
        self.filename = os.path.join('data', f'interview_qa_{data}_with_espn.tsv')
        self.use_wiki = use_wiki
        with open(self.filename, 'r') as r:
            lines = r.read()
            self.len = len([line for line in lines.split('\n') if len(line) > 3]) - 1
        del lines

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        line = linecache.getline(self.filename, item + 2).strip().split('\t')

        # espn
        if len(line[4]) > 0:
            all_news = []
            for news in line[4].split('|'):
                try:
                    with open(news) as f:
                        all_news.append(f.read())
                except:
                    file_path_rev = news[::-1]
                    folder_path = file_path_rev[file_path_rev.index('/'):][::-1]

                    file_name = None
                    best_score = 0
                    for news_title in os.scandir(folder_path):
                        matching = matching_score(news, news_title.path)
                        if matching > best_score:
                            file_name = news_title.path
                            best_score = matching

                    with open(file_name) as f:
                        all_news.append(f.read())

            line[4] = '|'.join(all_news)

        # wiki
        if self.use_wiki:
            # game wiki
            if len(line[2]) > 0:
                try:
                    with open(os.path.join('data', 'wiki', line[2])) as f:
                        line[2] = f.read()
                except FileNotFoundError:
                    # do nothing
                    line[2] = line[2]
                    print(line[2])
            # section wiki
            if len(line[3]) > 0:
                try:
                    with open(os.path.join('data', 'wiki', line[3])) as f:
                        line[3] = f.read()
                except FileNotFoundError:
                    line[3] = line[3]
                    print(line[3])
            # respondent wiki
            if len(line[13]) > 0:
                try:
                    with open(line[13]) as f:
                        line[13] = f.read()
                except FileNotFoundError:
                    line[13] = line[13]
                    print(line[13])

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return linecache.getline(self.filename, 1).strip().split('\t')


# sample usage of this class
if __name__ == '__main__':
    start_time = time.time()

    # dataset = InterviewDatasetESPN(use_wiki=True, data='train')
    #
    # data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)
    #
    # batch = next(iter(data_loader))
    # print(batch['question'])
    # print(batch['response'])
    # print(batch['respondent'])
    # print(batch['game_wiki_id'])
    # print(batch['section_wiki_id'])
    # print(batch['respondent_wiki'])
    # print(batch['prev_question'])
    # print(batch['prev_response'])

    respondent_set = set()
    utterance_count = 0
    word_count = 0
    import pysbd
    seg = pysbd.Segmenter(language='en', clean=False)
    input_len = []
    for file_type in ['train', 'dev', 'test']:
        dataset = InterviewDatasetESPN(use_wiki=True, data=file_type)

        data_loader = torch.utils.data.DataLoader(dataset, batch_size=1000, shuffle=False)

        for batch in tqdm(data_loader):
            # respondent_set.add(batch['respondent'][0])
            # utterance_count += len([sentence for sentence in list(seg.segment(batch['question'][0]))])
            # utterance_count += len([sentence for sentence in list(seg.segment(batch['response'][0]))])
            # word_count += len(batch['question'][0].split())
            # word_count += len(batch['response'][0].split())

            # batch_q = batch['question']
            # batch_r = batch['response']
            # batch_game_wiki = batch['game_wiki_id']
            # batch_section_wiki = batch['section_wiki_id']
            # batch_respondent_wiki = batch['respondent_wiki']
            # for i in range(len(batch_q)):
            #     if batch_game_wiki[i] != '':
            #         batch_game_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_game_wiki[i]))
            #     if batch_section_wiki[i] != '':
            #         batch_section_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_section_wiki[i]))
            #     if batch_respondent_wiki[i] != '':
            #         batch_respondent_wiki[i] = '. '.join(find_top_k(batch_q[i], batch_respondent_wiki[i]))
            #
            # batch_wiki = [f'{game_wiki.strip()}. {section_wiki.strip()}. {respondent_wiki.strip()}.'
            #               for game_wiki, section_wiki, respondent_wiki in
            #               zip(batch_game_wiki, batch_section_wiki, batch_respondent_wiki)]
            #
            # inputs = [len(wiki + q) + 1 for q, wiki in zip(batch_q, batch_wiki)]

            batch_q = batch['question']
            batch_r = batch['response']
            # batch_prev_q = batch['prev_question']
            # batch_prev_r = batch['prev_response']
            #
            # inputs = [len((prev_q + prev_r + q).split()) + 2
            #           for q, prev_q, prev_r in zip(batch_q, batch_prev_q, batch_prev_r)]
            inputs = [len(q.split()) for q in batch_q]
            input_len.extend(inputs)

    sns.set_theme()
    sns.set_context('paper')
    sns.histplot(input_len, binrange=(0, 100), binwidth=1)
    plt.xlabel('Response length (number of words)')
    plt.ylabel('Number of questions')
    # plt.show()
    plt.savefig(os.path.join('figures', 'bart-input-length-distribution.png'))


    # print(f'There are {len(respondent_set)} unique interviewees.')
    # print(f'There are {utterance_count} utterances.')
    # print(f'There are {word_count} words.')

    # counter = 0
    # total = 0
    # for batch in data_loader:
    #     if batch['espn'][0] == '':
    #         total += 0
    #     else:
    #         total += len(batch['espn'][0].split('|'))
    #     counter += 1
    # print(counter)
    # print(total)
    # print(total / counter)
    # print(f'Size of the dataset: {len(dataset)}')

    print(f'Time elapsed: {time.time() - start_time}')
