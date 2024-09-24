#!/bin/sh 
#SBATCH --job-name=llama7B_infr # Job name 
#SBATCH --time=23:59:00 # Time limit hrs:min:sec 
#SBATCH --output=llama7B_infr%j.out # Standard output and error log 
#SBATCH --partition=low_unl_1gpu

CUDA_VISIBLE_DEVICES=3 /data5/home/sahilm/anaconda3/bin/python /data5/home/sahilm/NLP_Project/llama_fine_tune/llam_evaluate_sriram.py