import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

tokenizer = AutoTokenizer.from_pretrained("Mistral-7B-v0.1")
model = AutoModelForCausalLM.from_pretrained(
    "Mistral-7B-v0.1/",
    trust_remote_code=True,
    device_map="auto",
    #load_in_8bit=True,# For 8 bit quantization,
)
model.eval()
#model = torch.compile(model, mode = "max-autotune", backend="inductor")

text = '''This is how LLM's generate some results on how LLMs generate absolute crap when it comes to Assamese/Mizo'''

device = 'cuda'
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

generated_ids = model.generate(**model_inputs, temperature=0.1, top_k=1, top_p=1.0, repetition_penalty=1.4, min_new_tokens=16, max_new_tokens=128, do_sample=True)
decoded = tokenizer.decode(generated_ids[0])

answer = decoded.split(text)[-1]
answer = answer.replace("<s>", "").replace("</s>", "").replace("<pad>", "")
print(answer)