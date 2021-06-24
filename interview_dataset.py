import torch
import os
import linecache
import time


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

    dataset = InterviewDatasetESPN(use_wiki=True, data='test')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(data_loader))
    # print(batch['question'])
    # print(batch['response'])
    # print(batch['respondent'])
    # print(batch['game_wiki_id'])
    # print(batch['section_wiki_id'])
    # print(batch['respondent_wiki'])
    # print(batch['prev_question'])
    # print(batch['prev_response'])

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
    print(f'Size of the dataset: {len(dataset)}')

    print(f'Time elapsed: {time.time() - start_time}')
