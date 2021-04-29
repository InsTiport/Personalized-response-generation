import torch
import torch.nn as nn
from transformers import BartModel

class Bart_wiki_model(nn.Module):
    def __init__(self):
        super().__init__()

        self.bart = BartModel.from_pretrained('facebook/bart-base')
        self.linear = nn.Linear(3 * self.bart.config.d_model, self.bart.config.vocab_size)

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, section_wiki_encoding, game_wiki_encoding):
        outputs = self.bart(input_ids=input_ids,
                            attention_mask=attention_mask,
                            decoder_input_ids=decoder_input_ids,
                            decoder_attention_mask=decoder_attention_mask)

        last_hidden_state = outputs.last_hidden_state

        out_seq_len = last_hidden_state.shape[1]
        batch_size = last_hidden_state.shape[0]

        section_wiki_encoding = section_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        game_wiki_encoding = game_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        with_wiki = torch.cat((last_hidden_state, section_wiki_encoding, game_wiki_encoding), dim=2)
        logits = self.linear(with_wiki)
        return logits