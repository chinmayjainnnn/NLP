from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("google/gemma-7b-it")
model = AutoModelForCausalLM.from_pretrained("google/gemma-7b-it", device_map="auto")


# Open the text file
with open('/raid/home/chinmayjain/NLP_project/gemma/Dataset/English.txt', 'r') as file:
    # Read lines from the file and strip any leading/trailing whitespace
    english_lines = [line.strip() for line in file.readlines()][5:10]

with open('/raid/home/chinmayjain/NLP_project/gemma/Dataset/Hindi.txt', 'r') as file:
    # Read lines from the file and strip any leading/trailing whitespace
    hindi_lines = [line.strip() for line in file.readlines()][5:10]


for line in english_lines:
    prompt ='translate the following text to hindi:'+ line 
    prompt = f'''<bos><start_of_turn>user 
{prompt}<end_of_turn>
<start_of_turn>model'''
    
    input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**input_ids,max_length=512)
    print(tokenizer.decode(outputs[0]))
    
# input_text = "translate the following text to hindi The brain is divided into two hemispheres, the left hemisphere and the right hemisphere. The left hemisphere is responsible for logical thinking, language processing, and mathematics. The right hemisphere is responsible for creative thinking, spatial awareness, and emotional processing."
# input_ids = tokenizer(input_text, return_tensors="pt").to("cuda")

# outputs = model.generate(**input_ids,max_length=512)
# print(tokenizer.decode(outputs[0]))
