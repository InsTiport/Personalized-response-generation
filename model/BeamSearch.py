import torch


def greedy_search(decoder, encoder_output, attention_mask, h, c, speaker_id=None, bos_token=0, eos_token=2,
                  max_length=100):
    cur_token = bos_token

    res = [bos_token]
    for _ in range(max_length):
        decoder_input = encoder_output.new_full((1, 1), cur_token)

        logits, h, c = decoder(decoder_input, h, c, encoder_output, attention_mask, speaker_id=speaker_id, train=False)
        # logits.shape: (1, vocab_size)

        # greedy decoding
        cur_token = torch.argmax(logits.squeeze()).item()
        res.append(cur_token)

        # exit if </s> is produced
        if cur_token == eos_token:
            break

    return res
