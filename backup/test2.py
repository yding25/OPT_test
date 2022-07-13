from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
model = GPT2LMHeadModel.from_pretrained("gpt2-xl")

prompt = "can a bowl be filled with water? answer: yes \n \
          can a fork be filled with water? answer: no  \n \
          can a knife be filled with water? answer:"
generated = tokenizer.encode(prompt)
context = torch.tensor([generated])

past_key_values = None

for i in range(100):
    output = model(context, past_key_values=past_key_values)
    past_key_values = output.past_key_values
    token = torch.argmax(output.logits[..., -1, :])

    context = token.unsqueeze(0)
    generated += [token.tolist()]

sequence = tokenizer.decode(generated)
sequence = sequence.split(".")[:-1]
print(sequence)
