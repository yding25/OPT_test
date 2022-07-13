from transformers import AutoModelForCausalLM, AutoTokenizer

checkpoint = 'facebook/opt-1.3b'
model = AutoModelForCausalLM.from_pretrained(checkpoint, device_map="auto", offload_folder='./offload_folder')
tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

prompt = "What time do most groceries close?"
inputs = tokenizer(prompt, return_tensors="pt")

output = model.generate(
                        inputs["input_ids"].to(0), \
                        max_length=100, \
                        do_sample=True, \
                        top_k=0, \
                        temperature=0.7 \
                        )
print(tokenizer.decode(output[0].tolist()))
