from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = 'facebook/opt-2.7b'
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", offload_folder='./offload_folder')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

prompt = "can a bowl be filled with water? answer: yes \n \
          can a fork be filled with water? answer: no  \n \
          can a knife be filled with water? answer:"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
                        inputs["input_ids"].to(0), \
                        max_length=100, \
                        do_sample=True, \
                        top_k=0, \
                        temperature=0.01 \
                        )
print(tokenizer.decode(output[0].tolist()))