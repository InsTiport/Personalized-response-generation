import torch
import torch.nn as nn
from transformers import BartModel
from transformers import BartForConditionalGeneration

class Bart_Wiki(BartForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        self.bart = BartModel(config)
        self.linear = nn.Linear(3 * self.bart.config.d_model, self.bart.config.vocab_size)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should either be in ``[0, ...,
            config.vocab_size]`` or -100 (see ``input_ids`` docstring). Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``.

        Returns:
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if labels is not None:
            if decoder_input_ids is None:
                decoder_input_ids = shift_tokens_right(
                    labels, self.config.pad_token_id, self.config.decoder_start_token_id
                )

        outputs = self.bart(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            head_mask=head_mask,
            decoder_head_mask=decoder_head_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        last_hidden_state = outputs.last_hidden_state
        out_seq_len = last_hidden_state.shape[1]
        batch_size = last_hidden_state.shape[0]
        section_wiki_encoding = section_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        game_wiki_encoding = game_wiki_encoding.unsqueeze(1).expand(batch_size, out_seq_len, self.bart.config.d_model).to(input_ids.device)
        with_wiki = torch.cat((last_hidden_state, section_wiki_encoding, game_wiki_encoding), dim=2)
        lm_logits = self.linear(with_wiki)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))

        if not return_dict:
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            decoder_hidden_states=outputs.decoder_hidden_states,
            decoder_attentions=outputs.decoder_attentions,
            cross_attentions=outputs.cross_attentions,
            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
            encoder_hidden_states=outputs.encoder_hidden_states,
            encoder_attentions=outputs.encoder_attentions,
        )


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