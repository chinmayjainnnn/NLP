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

model_name = "NousResearch/Llama-2-7b-hf" #Llama-2-7b-chat-hf
# new_model = "/raid/home/sriramd/llama-2-7b-translation"#"llama-2-7b-translation"

device_map='auto'

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
    cache_dir="/raid/home/chinmayjain/NLP_project/llama/models/Llama-2-7b-hf" #/raid/home/sriramd/Llama-2-7b-chat-hf
)
# model = PeftModel.from_pretrained(base_model, new_model)
# model = model.merge_and_unload()
# model = base_model

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
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
# validation_results = {"english": [], "hindi": [], "llama_generation": []}
# bleu_scores = []
print('Evaluating')
with open("/raid/home/chinmayjain/NLP_project/llama/chinmay/results/Prompt_1/hi_en_7b_hf.txt", 'w') as f:
    for i in tqdm(range(len(valid_dataset['validation']))):
        english = valid_dataset['validation'][i]['translation']['en']
        hindi = valid_dataset['validation'][i]['translation']['hi']
        prompt = f'''System: You are a Translator from English to Hindi:
User:                                     
Following the same format above from the examples, What is the Translation of the sentence given below. 
[Hindi]: {hindi} 
[English]: '''
        input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
        output = model.generate(inputs=input_ids, temperature=0.9, stopping_criteria=stopping_criteria, max_new_tokens=512, repetition_penalty=1.1)
        generated = tokenizer.decode(output[0])
        # print(f'GENERATED: {generated}', file=f)
        
        print(f'GENERATED: {generated}',file=f)
        print(f'[Actual English]: {english}',file=f)
        print("="*50,file=f)
        # validation_results['english'].append(english)
        # validation_results['hindi'].append(hindi)
        # validation_results['llama_generation'].append(generated)
        # start = generated.find("[/INST]") + len("[/INST]") + 1
        # end = generated.find("</s>")
        # if end == -1:
        #     prediction = generated[start:]
        # else:
        #     prediction = generated[start:end]
        # if prediction[-1] == '।':
        #     prediction = prediction[:-1]
        # hindi = hindi.replace("\'", "\"")
        # prediction = prediction.replace("\'", "\"")
        # # print(f'Actual Hindi: {hindi}\nPredicted: {prediction}',file=f)
        # bleu = metric.compute(predictions=[prediction], references=[[hindi]])
        # bleu_scores.append(bleu['score'])
            
    # print('BLEU Score: ', sum(bleu_scores)/len(bleu_scores))

    # df = pd.DataFrame(validation_results)
    # df.to_csv('validation_results.csv', index=False)