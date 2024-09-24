from datasets import load_dataset

with open("/data5/home/sahilm/NLP_Project/Dataset/English.txt",'r') as file1:
    english_sentences = file1.readlines()


with open("/data5/home/sahilm/NLP_Project/Dataset/Hindi.txt",'r') as file2:
    hindi_sentences = file2.readlines()


# print(len(english_sentences))
# print(len(hindi_sentences))

with open("/data5/home/sahilm/NLP_Project/Dataset/fine_tune_dataset.txt",'w') as file3:

    for i in range(len(english_sentences)):
        file3.write(f"""<s>[INST] Translate the following from English to Hindi: {english_sentences[i][:len(english_sentences[i])-1]} [/INST] {hindi_sentences[i][:len(hindi_sentences[i])-1]}</s>\n""")


en_hi_dataset = load_dataset("text", data_files={"train": ["/data5/home/sahilm/NLP_Project/Dataset/fine_tune_dataset.txt"]})