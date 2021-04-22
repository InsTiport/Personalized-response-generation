import torch
import os
import linecache
import time
import re
from pysbd import Segmenter
from sklearn.feature_extraction.text import TfidfVectorizer


def get_wiki(idx):
    with open(os.path.join('data', 'wiki', idx), 'r') as r:
        wiki = r.read()
    return wiki


def select_most_similar(query, text, top_l=5):
    text = re.sub(r'\n+', '\n', text)
    text = text.replace('\n', ' ')

    seg = Segmenter(language='en', clean=False)
    sentences = list(seg.segment(text))
    sentences = [s.strip() for s in sentences]

    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([query] + sentences).toarray()

    scores = vectors[1:, :] @ vectors[0].reshape(-1, 1).flatten()
    indices = scores.argsort()[-min(top_l, len(scores)):]

    # res is a string by concatenating all l most similar sentences
    res = ' '.join([sentences[1:][i] for i in range(len(sentences[1:])) if i in indices])

    return res


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

        if self.use_wiki:
            game_wiki = line[2]
            section_wiki = line[3]
            question = line[9]
            if game_wiki != '':
                line[2] = get_wiki(game_wiki)
                line[2] = select_most_similar(question, line[2])
            if section_wiki != '':
                line[3] = get_wiki(section_wiki)
                line[3] = select_most_similar(question, line[3])

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

        if self.use_wiki:
            game_wiki = line[2]
            section_wiki = line[3]
            question = line[9]
            if game_wiki != '':
                line[2] = get_wiki(game_wiki)
                line[2] = select_most_similar(question, line[2])
            if section_wiki != '':
                line[3] = get_wiki(section_wiki)
                line[3] = select_most_similar(question, line[3])

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return self.lines[0].strip().split('\t')


# sample usage of this class
if __name__ == '__main__':
    start_time = time.time()

    dataset = InterviewDataset(use_wiki=True)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    batch = next(iter(data_loader))
    print(batch['question'])
    print(batch['response'])
    print(batch['game_wiki'])
    print(batch['section_wiki'])

    print(f'Size of the dataset: {len(dataset)}')

    print(f'Time elapsed: {time.time() - start_time}')
