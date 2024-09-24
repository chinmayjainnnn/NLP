import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    pipeline,
)
from peft import PeftModel
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import StoppingCriteria, StoppingCriteriaList

# metric = evaluate.load("sacrebleu")

access_token ="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"

model_name = "sarvamai/OpenHathi-7B-Hi-v0.1-Base"

device_map='cuda:0'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    cache_dir="/data5/home/sahilm/Sriram/Nous",
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/data5/home/sahilm/Sriram/Nous", trust_remote_code=True, token = access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def create_stopping_criteria(stop_words, tokenizer, device):

    class StoppingCriteriaSub(StoppingCriteria):
        def __init__(self, stops = [], device=device, encounters = 1):
            super().__init__()
            self.stops = stops = [stop.to(device) for stop in stops]

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
            last_token = input_ids[0][-1]
            for stop in self.stops:
                if tokenizer.decode(stop) == tokenizer.decode(last_token):
                    return True
            return False

    stop_word_ids = [tokenizer(stop_word,
                               return_tensors="pt", 
                               add_special_tokens=False)["input_ids"].squeeze() 
                               for stop_word in stop_words]

    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_word_ids)])
    return stopping_criteria

stop_words_list = [">"]
stopping_criteria = None
if stop_words_list is not None:
    stopping_criteria = create_stopping_criteria(stop_words_list, tokenizer, "cuda")


#Validate on the validation set
valid_dataset = load_dataset("cfilt/iitb-english-hindi")

with open("/data5/home/sahilm/NLP_Project/open_hathi/open_hathi_hin_to_eng.txt",'w') as f:
    # validation_results = {"english": [], "hindi": [], "llama_generation": []}
    # bleu_scores = []
    print('Evaluating')
    print(len(valid_dataset['validation']))
    for i in tqdm(range(len(valid_dataset['validation']))):#tqdm(range(len(valid_dataset['validation']))):
        english = valid_dataset['validation'][i]['translation']['en']
        hindi = valid_dataset['validation'][i]['translation']['hi']
        # prompt = f'<s>[INST] Translate the following from English to Hindi: {english} [/INST]'
        prompt = f'''Translate from Hindi to English no extra text
[Hindi]: {hindi}
[English]:'''
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.9, stopping_criteria=stopping_criteria, max_new_tokens=512, repetition_penalty=1.1)
        generated = tokenizer.decode(output[0])

        print(generated,file=f)
        # print(generated)
        print(f'[Actual English]: {english}',file=f)
        # print(f'[Actual English]: {english}')
        print('='*50,file=f)
        # print('='*50)
        