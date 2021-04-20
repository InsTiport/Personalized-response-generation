from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained('gpt2')

generated = tokenizer("The Manhattan bridge")['input_ids']
context = torch.tensor([generated])
past = None

for i in range(100):
    print(i)
    out = model(context, past_key_values=past)
    output, past = out.logits, out.past_key_values
    token = torch.argmax(output[..., -1, :])

    generated += [token.tolist()]
    context = token.unsqueeze(0).unsqueeze(0)
    print(context.shape)

sequence = tokenizer.decode(generated)

print(sequence)
