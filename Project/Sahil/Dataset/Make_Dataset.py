from datasets import load_dataset

# dataset = load_dataset("ai4bharat/samanantar","hi",trust_remote_code=True)

with open('/data5/home/sahilm/NLP_Project/Dataset/Dest/final_data/en-hi/train.hi', 'r') as f1:
    lines = [f1.readline() for _ in range(10000)]

with open("/data5/home/sahilm/NLP_Project/Dataset/Hindi.txt", "w") as output_file:
    # Write the lines to the output file
    for line in lines:
        output_file.write(line)

with open('/data5/home/sahilm/NLP_Project/Dataset/Dest/final_data/en-hi/train.en', 'r') as f1:
    lines = [f1.readline() for _ in range(10000)]

with open("/data5/home/sahilm/NLP_Project/Dataset/English.txt", "w") as output_file:
    # Write the lines to the output file
    for line in lines:
        output_file.write(line)

# print(data)

# total = 5000
# with open('/data5/home/sahilm/NLP_Project/Dataset/English.txt', 'w') as file:
#     for i in range(total):
#         txt = dataset['train'][i]['src']
#         file.write(txt+'\n')
#         # print(dataset['train'][i]['src'])
#         # print(dataset['train'][i]['tgt'])

# with open('/data5/home/sahilm/NLP_Project/Dataset/Hindi.txt', 'w') as file:
#     for i in range(total):
#         file.write(dataset['train'][i]['tgt']+'\n')
#         # print(dataset['train'][i]['src'])
#         # print(dataset['train'][i]['tgt'])