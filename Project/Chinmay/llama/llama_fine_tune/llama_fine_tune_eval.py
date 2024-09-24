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

metric = evaluate.load("sacrebleu")

access_token ="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"

model_name = "NousResearch/Llama-2-7b-chat-hf"
new_model = "llama-2-7b-translation-3-epochs-10K-Sahil"

device_map='auto'

base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    # low_cpu_mem_usage=True,
    cache_dir="/raid/home/chinmayjain/NLP_project/llama_fine_tune/Nous",
    return_dict=True,
    torch_dtype=torch.float16,
    device_map=device_map,
)
model = PeftModel.from_pretrained(base_model, new_model)
model = model.merge_and_unload()
# model = base_model

# Reload tokenizer to save it
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir="/raid/home/chinmayjain/NLP_project/llama_fine_tune/Nous", trust_remote_code=True, token = access_token)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

def create_stopping_criteria(stop_words, tokenizer, device):

    class StoppingCriteriaSub(StoppingCriteria):
        def _init_(self, stops = [], device=device, encounters = 1):
            super()._init_()
            self.stops = stops = [stop.to(device) for stop in stops]

        def _call_(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> bool:
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
validation_results = {"english": [], "hindi": [], "llama_generation": []}
bleu_scores = []
print('Evaluating')
for i in tqdm(range(20)):#tqdm(range(len(valid_dataset['validation']))):
    english = valid_dataset['validation'][i]['translation']['en']
    # english = input("Enter English: ")
    hindi = valid_dataset['validation'][i]['translation']['hi']
    # prompt = f'<s>[INST] Translate the following from English to Hindi: {english} [/INST]'
    prompt = f'<s>[INST] Translate the following from English to Hindi: {english} [/INST]'
    
    input_ids = tokenizer(prompt, return_tensors='pt', add_special_tokens=False).input_ids.cuda()
    # output = model.generate(inputs=input_ids, do_sample=True, temperature=0.1, top_p=0.9, stopping_criteria=stopping_criteria, max_new_tokens=512, repetition_penalty=1.2)
    output = model.generate(inputs=input_ids, do_sample=True, temperature=0.1, top_p=0.9, max_new_tokens=512, repetition_penalty=1.2)
    generated = tokenizer.decode(output[0])
    print("Generated: ",generated)
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
    # print(f'Actual Hindi: {hindi}')
    # print(f'Predicted: {prediction}')
    # bleu = metric.compute(predictions=[prediction], references=[[hindi]])
    # bleu_scores.append(bleu['score'])
        
# print('BLEU Score: ', sum(bleu_scores)/len(bleu_scores))

# df = pd.DataFrame(validation_results)
# df.to_csv('validation_results.csv', index=False)