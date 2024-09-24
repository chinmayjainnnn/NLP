#!/bin/sh
#SBATCH --job-name=train_llama_full # Job name
#SBATCH --time=23:59:59 # Time limit hrs:min:sec
#SBATCH --output=full_training%j.out # Standard output and error log
#SBATCH --gres=gpu:2
CUDA_VISIBLE_DEVICES=5,7 python3 llama_full_training.py