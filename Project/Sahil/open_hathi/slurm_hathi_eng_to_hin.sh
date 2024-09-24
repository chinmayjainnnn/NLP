#!/bin/sh 
#SBATCH --job-name=Open_hathi_eng_to_hin # Job name 
#SBATCH --time=23:59:00 # Time limit hrs:min:sec 
#SBATCH --output=Open_hathi_eng_to_hin%j.out # Standard output and error log 
#SBATCH --gres=gpu:1 
#SBATCH --partition=low_unl_1gpu

/data5/home/sahilm/anaconda3/bin/python /data5/home/sahilm/NLP_Project/open_hathi/inference_eng_to_hin.py