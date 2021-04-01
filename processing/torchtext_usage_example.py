from torchtext.data import TabularDataset, BucketIterator, RawField
import os

BATCH_SIZE = 10

os.chdir('../')

question = RawField()
response = RawField()
fields = {'question': ('q', question), 'response': ('r', response)}

dataset = TabularDataset(path=os.path.join('data', 'csv', 'single_turn_utterance.csv'), format='csv', fields=fields)

iterator = BucketIterator(dataset=dataset, batch_size=BATCH_SIZE)

for batch in iterator:
    # sample usage
    print(batch.q)
    print(batch.r)
    break
