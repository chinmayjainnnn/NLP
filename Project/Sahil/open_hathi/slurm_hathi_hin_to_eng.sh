#!/bin/sh 
#SBATCH --job-name=Open_hathi_hin_to_eng # Job name 
#SBATCH --time=23:59:00 # Time limit hrs:min:sec 
#SBATCH --output=Open_hathi_hin_to_eng%j.out # Standard output and error log 
#SBATCH --gres=gpu:2 
#SBATCH --partition=low_unl_1gpu

/data5/home/sahilm/anaconda3/bin/python /data5/home/sahilm/NLP_Project/open_hathi/inference_hin_to_eng.py