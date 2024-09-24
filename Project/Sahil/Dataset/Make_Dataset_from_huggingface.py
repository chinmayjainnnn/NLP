from datasets import load_dataset

dataset = load_dataset("ai4bharat/samanantar","hi", cache_dir="/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset")

total = 10000
with open('/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/English.txt', 'w') as file:
    for i in range(total):
        file.write(dataset['train'][i]['src']+'\n')
        # print(dataset['train'][i]['src'])
        # print(dataset['train'][i]['tgt'])

with open('/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/Hindi.txt', 'w') as file:
    for i in range(total):
        file.write(dataset['train'][i]['tgt']+'\n')
        # print(dataset['train'][i]['src'])
        # print(dataset['train'][i]['tgt'])