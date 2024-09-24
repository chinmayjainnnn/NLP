from transformers import AutoModelForCausalLM, AutoTokenizer
import transformers
import torch
from accelerate import Accelerator

model_path = "meta-llama/Llama-2-7b-chat-hf"

model = AutoModelForCausalLM.from_pretrained(
    model_path,    
    cache_dir="/data5/home/sahilm/NLP_Project/Llama_2_7b_chat_hf",
)
print("Valid 1")

tokenizer = AutoTokenizer.from_pretrained(model_path, cache_dir="/data5/home/sahilm/NLP_Project/Llama_2_7b_chat_hf")
print("Valid 2")

# Initialize the Accelerator
accelerator = Accelerator()

model = accelerator.prepare(model)
print("Valid 3")

# Create the text generation pipeline
def generate_text(text):
    sequences = accelerator.split_batch(pipeline, text, num_processes=len(accelerator.devices))
    return sequences

pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto",
)

# # Generate text
# text = 'I liked "Breaking Bad" and "Band of Brothers". Do you have any recommendations of other shows I might like?\n'
# sequences = generate_text(text)

# # Gather results from all devices
# all_sequences = accelerator.gather(sequences)

# # Rebuild the final result
# final_result = []
# for seq in all_sequences:
#     final_result.append(seq['generated_text'])

# # Print the final result
# for result in final_result:
#     print(f"Result: {result}")
