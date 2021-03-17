import os
import datasets
import numpy as np
import torch
from torchtext.data import TabularDataset, BucketIterator, RawField
from transformers import BartForConditionalGeneration, BartTokenizer

os.chdir('../')

'''
hyper-parameter 
'''
DEVICE_ID = 3  # adjust this to use an unoccupied GPU
EVAL_BATCH_SIZE = 2
MODEL_NAME = f'bart-base_epoch_2_bsz_3_small_utterance'
# sampling specifications
TOP_P = 0.92
TOP_K = 0
NUM_RETURN_SENTENCES = 1

'''
load dataset
'''
# prepare fields (needed when loading dataset)
question = RawField()
response = RawField()
fields = {'question': ('q', question), 'response': ('r', response)}
# load dataset
train_set, valid_set, test_set = TabularDataset.splits(path=os.path.join('data', 'csv'),
                                                       train='smaller_utterance_train.csv',
                                                       validation='smaller_utterance_valid.csv',
                                                       test='smaller_utterance_test.csv',
                                                       format='csv',
                                                       fields=fields)

# split dataset into batches
train_iterator = BucketIterator(dataset=train_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)
valid_iterator = BucketIterator(dataset=valid_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)
test_iterator = BucketIterator(dataset=test_set, batch_size=EVAL_BATCH_SIZE, shuffle=True)

'''
model and tokenizer
'''
# CUDA settings
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if device == 'cuda':
    torch.cuda.set_device(DEVICE_ID)  # use an unoccupied GPU
# load model
SAVE_PATH = os.path.join('model', f'{MODEL_NAME}.pt')
model = BartForConditionalGeneration.from_pretrained('facebook/bart-base').to(device)
model.load_state_dict(torch.load(SAVE_PATH))
model.eval()
# load tokenizer
tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')

'''
compute BLEU and perplexity in validation set
'''
metric = datasets.load_metric('sacrebleu')
with torch.no_grad():
    batch_num = 0
    perplexity_sum = 0
    for batch in valid_iterator:
        # FIXME for now, skip all invalid question-answer pairs (those having questions longer than 685)
        remove_idx = [i for i, q in enumerate(batch.q) if len(q) >= 685]
        batch_q = [q for i, q in enumerate(batch.q) if i not in remove_idx]
        batch_r = [r for i, r in enumerate(batch.r) if i not in remove_idx]
        assert len(batch_q) == len(batch_r)
        if len(batch_q) == 0:
            continue

        # input encoding
        input_encoding = tokenizer(batch_q, return_tensors='pt', padding=True, truncation=True)
        input_ids = input_encoding['input_ids'].to(device)
        attention_mask = input_encoding['attention_mask'].to(device)

        # target encoding
        target_encoding = tokenizer(batch_r, return_tensors='pt', padding=True, truncation=True)
        target_ids = target_encoding['input_ids'].to(device)
        target_ids[target_ids == model.config.pad_token_id] = -100

        # forward pass
        outputs = model(input_ids, attention_mask=attention_mask, labels=target_ids)

        # loss
        loss = outputs.loss

        # record perplexity
        perplexity_sum += np.exp(loss.item())

        # generation
        model_res_ids = model.generate(input_ids,
                                       do_sample=True,
                                       max_length=model.config.max_position_embeddings,
                                       top_p=TOP_P,
                                       top_k=TOP_K,
                                       num_return_sequences=NUM_RETURN_SENTENCES)
        # add generated responses and gold responses for future BLEU computation
        predictions = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=False) for g in
                       model_res_ids]
        references = [[r] for r in batch_r]
        metric.add_batch(predictions=predictions, references=references)

        batch_num += 1

    # BLEU
    score = metric.compute()
    # ppl
    perplexity = perplexity_sum / batch_num

    print(f'Perplexity: {perplexity}')
    print(f'BLEU: {round(score["score"], 1)} our of {round(100., 1)}')

# # sample predictions which get full BLEU score
# predictions = ["hello there general kenobi", "foo bar foobar"]
# references = [["hello there general kenobi", "hello there !"],
#               ["foo bar foobar", "foo bar foobar"]]
# results = metric.compute(predictions=predictions, references=references)
# print(round(results["score"], 1))
