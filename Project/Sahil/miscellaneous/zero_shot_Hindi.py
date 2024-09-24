#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel
import numpy as np
import os
import re
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"    # Free GPU
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")


# In[ ]:


model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-7b-chat-hf",
    cache_dir="/data5/home/sahilm/Sriram/Nous"
    )

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf", 
                                          cache_dir="/data5/home/sahilm/Sriram/Nous"
                                         )


# In[ ]:


if torch.cuda.device_count() > 1:
    model = DataParallel(model)
    print("Moj")
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


# In[53]:


model_without_parallel = model.module


# In[113]:


from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize

def calculate_bleu_score(reference, candidate):
    # Tokenize reference and candidate strings
    reference_tokens = word_tokenize(reference.lower())
    candidate_tokens = word_tokenize(candidate.lower())
    # print(reference_tokens)
    # print([reference_tokens])
    # print(candidate_tokens)
    # Calculate BLEU score
    bleu_score = sentence_bleu([reference_tokens], candidate_tokens,smoothing_function=SmoothingFunction().method1)
    
    return bleu_score

# Example usage:
# reference = "मैं क्रिकेट खेलना चाहता हूं"
# candidate = "मैं क्रिकेट खेलना चाहता हूं"
# score = calculate_bleu_score(reference, candidate)
# print("BLEU Score:", score)


# In[89]:


english_file = '/data5/home/sahilm/NLP_Project/Dataset/Dest/final_data/en-hi/train.en'
gujarati_file = '/data5/home/sahilm/NLP_Project/Dataset/Dest/final_data/en-hi/train.hi'

english_sentences = []
gujarati_sentences = []
total_sentences = 5000

with open(english_file, 'r') as file:
    for i, line in enumerate(file):
        if i >= total_sentences:
            break
        english_sentences.append(line)

with open(gujarati_file, 'r') as file:
    for i, line in enumerate(file):
        if i >= total_sentences:
            break
        gujarati_sentences.append(line)


print(len(english_sentences))
print(len(gujarati_sentences))

# total_sentences = 5000
# english_sentences = english_sentences[:total_sentences]
# gujarati_sentences = gujarati_sentences[:total_sentences]

english_sentences = [sentence.rstrip('\n') for sentence in english_sentences]
gujarati_sentences = [sentence.rstrip('\n') for sentence in gujarati_sentences]




# In[90]:


# for i in range(3):
#     print(english_sentences[i])
#     print(gujarati_sentences[i])
#     print()


# In[91]:


PERCENTILE = 97
# print(np.percentile([len(x) for x in english_sentences], PERCENTILE))
# print(np.percentile([len(x) for x in gujarati_sentences], PERCENTILE))


# In[92]:


max_seq_length = 300


# In[93]:


# for i in range(3):
#     print(english_sentences[i])
#     print(gujarati_sentences[i])
#     print()


# In[94]:


from torch.utils.data import Dataset, DataLoader
class TextDataSet(Dataset):
    def __init__(self, english_sentences, gujarati_sentences):
        super().__init__
        self.english_sentences = english_sentences
        self.gujarati_sentences = gujarati_sentences
        
    def __len__(self):
        return len(self.english_sentences)
    
    def __getitem__(self,idx):
        return self.english_sentences[idx], self.gujarati_sentences[idx]
dataset = TextDataSet(english_sentences,gujarati_sentences)  


# In[95]:


# len(dataset)


# In[96]:


# dataset[22]


# In[97]:


batch_size=150
train_loader = DataLoader(dataset,batch_size)
iterator = iter(train_loader)


# In[98]:


# for batch in train_loader:
#     # Do something with the batch
#     # print(batch)
#     # print(len(batch))
#     # print(len(batch[0]))
#     # print(len(batch[1]))
#     for i in range(len(batch[0])):
#         print(batch[0][i])
#         print(batch[1][i])
#         print()
#         if i==2:
#             break
#     break


# In[102]:


import transformers

pipeline = transformers.pipeline(
    "text-generation",
    model = model_without_parallel,
    tokenizer = tokenizer,
    torch_dtype = torch.float16,
    device = 0 if device.type == "cuda" else -1,
    batch_size = 128,
    truncation=True
)


# In[106]:


total_bleu_score = 0
max_bleu_score = 0
with open("zero_shot_Hindi_output_IITB.txt", 'w') as f:

    for batch in train_loader:
        # Do something with the batch
        # print(batch)
        # print(len(batch))
        # print(len(batch[0]))
        # print(len(batch[1]))
        for i in range(len(batch[0])):
            text = batch[0][i].strip()
            # text = "I want to play cricket"
            
            reference = batch[1][i].strip()
            # reference = "मैं क्रिकेट खेलना चाहता हूं"
            
            print("ENGLISH: " + text)
            print("ENGLISH: " + text, file=f)
            print("HINDI: " + reference)
            print("HINDI: " + reference, file=f)
            
            template = f"""Translate the following text to Hindi:\nText: {text}\nOutput: """    
            
            sequences = pipeline(
                template,
                do_sample = True,
                top_k = 10,
                num_return_sequences = 1,
                eos_token_id = tokenizer.eos_token_id,
                max_length = max_seq_length,
                truncation = True
            )
            output = sequences[0]['generated_text']

            sentences_after_output = re.findall(r'Output:(.*)', output)
            output = sentences_after_output[0].strip()
            print("OUTPUT: " + output)
            print("OUTPUT: " + output,file=f)
            bleu_score = calculate_bleu_score(reference, output)
            max_bleu_score = max(bleu_score,max_bleu_score)
            total_bleu_score += bleu_score
            print("BLEU Score:", bleu_score)
            print("BLEU Score:", bleu_score,file=f)
            print("Average BLEU Score:", total_bleu_score/(i+1))
            print("Average BLEU Score:", total_bleu_score/(i+1),file=f)
            print()
            print("",file=f)

    print("Max BLEU Score: ", max_bleu_score)
    print("Max BLEU Score: ", max_bleu_score,file=f)


# In[ ]:




