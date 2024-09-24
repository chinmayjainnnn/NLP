#!/bin/sh 
#SBATCH --job-name=FINE_TUNE_llama_7B # Job name 
#SBATCH --time=23:59:00 # Time limit hrs:min:sec 
#SBATCH --output=FINE_TUNE_llama_7B%j.out # Standard output and error log 
#SBATCH --gres=gpu:2
#SBATCH --partition=med_24h_4gpu

CUDA_VISIBLE_DEVICES=1,6 /data5/home/sahilm/anaconda3/bin/python /data5/home/sahilm/NLP_Project/llama_fine_tune/llama_7B.py