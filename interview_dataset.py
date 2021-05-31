import torch
import os
import linecache
import time


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


class InterviewDatasetAlternatives(torch.utils.data.Dataset):
    def __init__(self, data='train', use_wiki=False):
        self.filename = os.path.join('data', f'interview_qa_{data}.tsv')
        self.use_wiki = use_wiki
        with open(self.filename, 'r') as r:
            self.lines = r.readlines()
            self.len = len([line for line in self.lines if len(line) > 3]) - 1

    def __len__(self):
        return self.len

    def __getitem__(self, item):
        line = self.lines[item + 1].strip().split('\t')

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return self.lines[0].strip().split('\t')


# sample usage of this class
if __name__ == '__main__':
    start_time = time.time()

    dataset = InterviewDatasetWithPrevQR(use_wiki=True, data='dev')

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

    batch = next(iter(data_loader))
    print(batch['question'])
    print(batch['response'])
    # print(batch['game_wiki'])
    # print(batch['section_wiki'])
    print(batch['prev_question'])
    print(batch['prev_response'])

    print(f'Size of the dataset: {len(dataset)}')

    print(f'Time elapsed: {time.time() - start_time}')
