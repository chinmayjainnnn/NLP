from datasets import load_dataset

with open("/data5/home/sahilm/NLP_Project/Dataset/instruction_dataset/English.txt",'r') as file1:
    english_sentences = file1.readlines()


with open("/data5/home/sahilm/NLP_Project/Dataset/instruction_dataset/Hindi.txt",'r') as file2:
    hindi_sentences = file2.readlines()


# print(len(english_sentences))
# print(len(hindi_sentences))

with open("/data5/home/sahilm/NLP_Project/Dataset/instruction_dataset/fine_tune_instr_dataset.txt",'w') as file3:

    for i in range(len(english_sentences)):
        file3.write(f"""Below is an instruction that describes a task. ### Instruction: Translate the following from English to Hindi: {english_sentences[i][:len(english_sentences[i])-1]} [/INST] {hindi_sentences[i][:len(hindi_sentences[i])-1]}</s>\n""")


en_hi_dataset = load_dataset("text", data_files={"train": ["/data5/home/sahilm/NLP_Project/Dataset/instruction_dataset/fine_tune_instr_dataset.txt"]})