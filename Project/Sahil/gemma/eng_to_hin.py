# pip install accelerate
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

access_token ="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"
tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it",cache_dir="/raid/home/chinmayjain/Sahil/NLP_Project/gemma/model",token = access_token)
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it",cache_dir="/raid/home/chinmayjain/Sahil/NLP_Project/gemma/model", device_map="auto",token = access_token)


english_file = '/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/English.txt'
hindi_file = '/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/Hindi.txt'
with open(english_file,'r') as file:
    english_sentences = file.readlines()
with open(hindi_file,'r') as file:
    hindi_sentences = file.readlines()


output_file = "/raid/home/chinmayjain/Sahil/NLP_Project/gemma/output_eng_to_hin_gemma.txt"

with open(output_file, 'w') as f:
    for english_sentence in tqdm(english_sentences):    # prompt = "What is your favorite condiment?"
        # prompt = input('Enter Prompt: ')
        prompt = f"""translate the following text from english to hindi script '{english_sentence[:len(english_sentence)-1]}'"""
        input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")

        outputs = model.generate(**input_ids,max_length=2048,top_k=5,temperature=0.1,do_sample=True)
        # print(tokenizer.batch_decode(outputs, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0])
        print(tokenizer.decode(outputs[0],skip_special_tokens=True, clean_up_tokenization_spaces=False),file=f)
        print("\n=============================\n",file=f)
