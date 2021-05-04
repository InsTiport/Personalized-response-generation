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
            decoder_attention_mask=decoder_attention_mask
        )

        last_hidden_state = outputs.last_hidden_state

        out_seq_len = last_hidden_state.shape[1]
        batch_size = last_hidden_state.shape[0]

        section_wiki_encoding = section_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        game_wiki_encoding = game_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        with_wiki = torch.cat((last_hidden_state, section_wiki_encoding, game_wiki_encoding), dim=2)
        logits = self.linear(with_wiki)
        return logits

    def decode_greedy(self, input_ids, attention_mask, section_wiki_encoding, game_wiki_encoding, max_length=100):
        section_wiki_encoding = section_wiki_encoding.unsqueeze(1).to(input_ids.device)
        game_wiki_encoding = game_wiki_encoding.unsqueeze(1).to(input_ids.device)

        output_attentions = self.bart.config.output_attentions
        output_hidden_states = self.bart.config.output_hidden_states
        use_cache = self.bart.config.use_cache
        return_dict = self.bart.config.use_return_dict


        encoder_outputs = self.bart.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask, 
            return_dict=return_dict
        )

        curr_token = self.bart.config.bos_token_id
        res = [curr_token]
        for _ in range(max_length):
            decoder_input_ids = torch.tensor(curr_token, dtype=torch.long).unsqueeze(0).repeat(input_ids.shape[0], 1).to(input_ids.device)

            decoder_outputs = self.bart.decoder(
                input_ids=decoder_input_ids,
                encoder_hidden_states=encoder_outputs[0],
                encoder_attention_mask=attention_mask,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict
            )

            last_hidden_state = decoder_outputs.last_hidden_state
            out_seq_len = last_hidden_state.shape[1]
            batch_size = last_hidden_state.shape[0]
            with_wiki = torch.cat((last_hidden_state, section_wiki_encoding, game_wiki_encoding), dim=2)
            logits = self.linear(with_wiki)

            curr_token = torch.argmax(logits.squeeze()).item()
            res.append(curr_token)

            if curr_token == self.bart.config.eos_token_id:
                break

        return res