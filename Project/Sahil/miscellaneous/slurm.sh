#!/bin/sh 
#SBATCH --job-name=llama_7B # Job name 
#SBATCH --time=23:59:00 # Time limit hrs:min:sec 
#SBATCH --output=llama_7B%j.out # Standard output and error log 
#SBATCH --partition=med_24h_4gpu

CUDA_VISIBLE_DEVICES=1,2 /data5/home/sahilm/anaconda3/bin/python /data5/home/sahilm/NLP_Project/llama_fine_tune/llama_7B.py