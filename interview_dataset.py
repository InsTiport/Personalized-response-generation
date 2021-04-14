import torch
import os
import linecache
import time


def get_wiki(idx):
    with open(os.path.join('data', 'wiki', idx), 'r') as r:
        wiki = r.read()
    return wiki


class InterviewDataset(torch.utils.data.Dataset):
    def __init__(self, use_wiki=False):
        self.filename = os.path.join('data', 'interview_qa.tsv')
        self.use_wiki = use_wiki

    def __len__(self):
        return 1759312

    def __getitem__(self, item):
        line = linecache.getline(self.filename, item + 2).strip().split('\t')

        if self.use_wiki:
            game_wiki = line[2]
            section_wiki = line[3]
            if game_wiki != '':
                line[2] = get_wiki(game_wiki)
            if section_wiki != '':
                line[3] = get_wiki(section_wiki)

        return {k: v for k, v in zip(self._get_header(), line)}

    def _get_header(self):
        return linecache.getline(self.filename, 1).strip().split('\t')


# sample usage of this class
if __name__ == '__main__':
    start_time = time.time()

    dataset = InterviewDataset(use_wiki=False)

    data_loader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True, num_workers=0)

    batch = next(data_loader)
    print(batch)

    print(f'Time elapsed: {time.time() - start_time}')
