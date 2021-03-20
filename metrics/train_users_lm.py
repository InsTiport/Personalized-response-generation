import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config, AdamW
from transformers import AdamW
import os
import tqdm
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class GPT2Dataset(Dataset):

    def __init__(self, data_file, tokenizer, gpt2_type="gpt2", max_length=512):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []

        with open(data_file, 'r') as f:
            for line in f:
                encodings_dict = tokenizer('<|startoftext|>'+ line + '<|endoftext|>', truncation=True, max_length=max_length, padding="max_length")
                self.input_ids.append(torch.tensor(encodings_dict['input_ids']))
                self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.attn_masks[idx]

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument()
arg_parser.add_argument(
    '--batch_size',
    type=int,
    default=3,
    help=f'Specify evaluation batch size'
)
arg_parser.add_argument(
    '--num_epochs',
    type=int,
    default=5,
    help=f'Specify the number of epochs for training'
)

args = arg_parser.parse_args()
batch_size = args.batch_size
NUM_EPOCH = args.num_epochs

learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

data_path = os.path.join('data', 'csv', 'smaller_utterance_train.csv')
language_models = set()

tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>')

os.chdir('../')
os.makedirs(os.path.join('model', 'user_lm'), exist_ok=True)
user = 0
for user_file in os.scandir(os.path.join('data', 'user_corpus')):
    # TODO: fine tune GPT2 for each user
    dataset = GPT2Dataset(os.path.join('data', 'user_corpus', user_file.name), tokenizer)
    
    # Split into training and validation sets
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
    validation_dataloader = DataLoader(val_dataset, batch_size=batch_size)

    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
    model = GPT2LMHeadModel.from_pretrained('gpt2', config=configuration)
    model.resize_token_embeddings(len(tokenizer))

    device = torch.device('cuda')
    model = model.to(device)

    optimizer = AdamW(model.parameters(), lr = learning_rate, eps = epsilon)
    total_steps = len(train_dataloader) * NUM_EPOCH
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warmup_steps, num_training_steps = total_steps)

    for epoch in range(NUM_EPOCH):
        print("")
        print(f'======== Epoch {epoch + 1} / {NUM_EPOCH} (user {user}) ========')
        print('Training...')

        total_train_loss = 0
        model.train()
        for step, batch in tqdm.tqdm(enumerate(train_dataloader)):
            input_ids = batch[0].to(device)
            labels = batch[0].to(device)
            masks = batch[1].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, labels=labels, attention_mask=masks, token_type_ids=None)

            loss = outputs[0]
            total_train_loss += loss.item()

            loss.backward()
            optimizer.step()
    torch.save(model.state_dict(), os.path.join('model', 'user_lm', f'gpt2_lm_{user}.pt'))

    avg_train_loss = total_train_loss / len(train_dataloader)
    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    user += 1