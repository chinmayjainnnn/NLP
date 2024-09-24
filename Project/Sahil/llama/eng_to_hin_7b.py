# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

import fire
from tqdm import tqdm
import torch.distributed as dist

from llama import Llama
from typing import List

def main(
    ckpt_dir: str,
    tokenizer_path: str,
    temperature: float = 0.0,
    top_p: float = 0.9,
    max_seq_len: int = 512,
    max_gen_len: int = 512,
    max_batch_size: int = 4,
):
    """
    Entry point of the program for generating text using a pretrained model.

    Args:
        ckpt_dir (str): The directory containing checkpoint files for the pretrained model.
        tokenizer_path (str): The path to the tokenizer model used for text encoding/decoding.
        temperature (float, optional): The temperature value for controlling randomness in generation.
            Defaults to 0.6.
        top_p (float, optional): The top-p sampling parameter for controlling diversity in generation.
            Defaults to 0.9.
        max_seq_len (int, optional): The maximum sequence length for input prompts. Defaults to 128.
        max_gen_len (int, optional): The maximum length of generated sequences. Defaults to 64.
        max_batch_size (int, optional): The maximum batch size for generating sequences. Defaults to 4.
    """ 
    dist.init_process_group(backend="nccl")

    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size,
    )

    # english_file = '/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/English.txt'
    # hindi_file = '/raid/home/chinmayjain/Sahil/NLP_Project/llama/Dataset/Hindi.txt'
    # with open(english_file,'r') as file:
    #     english_sentences = file.readlines()
    # with open(hindi_file,'r') as file:
    #     hindi_sentences = file.readlines()

    #Upadte
    from datasets import load_dataset
    valid_dataset = load_dataset("cfilt/iitb-english-hindi")
    english_sentences = [valid_dataset['validation'][i]['translation']['en'] for i in range(len(valid_dataset['validation']))]
    hindi_sentences = [valid_dataset['validation'][i]['translation']['hi'] for i in range(len(valid_dataset['validation']))]


    output_file = "/data5/home/sahilm/NLP_Project/llama/output_eng_to_hin_7b.txt"
    with open(output_file, 'w') as f:
        for english_sentence in tqdm(english_sentences):
    # while True:
            # english_sentence = input("Enter an English sentence to translate to Hindi (type 'exit' to quit): ")
            # prompt = f"""I am giving you one sentence Translate this sentence from English to Hindi\n[English]: {english_sentence}\n[Hindi]:"""
            prompt = f"""[English]: {english_sentence}\nTranslate from [English] to [Hindi]:"""
            if english_sentence.lower() == 'exit':
                print("Exiting translation program.")
                break
            
            result = generator.text_completion([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p)[0]
            # print(prompt)
            print(prompt,file=f)
            # print(f"Translated Hindi: {result['generation']}")
            print(f"Translated Hindi: {result['generation']}",file=f)
            # print("\n==================================\n")
            print("\n==================================\n",file=f)


if __name__ == "__main__":
    fire.Fire(main)
