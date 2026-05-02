#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=24:00:00
#SBATCH --job-name=gec_train
#SBATCH --output=train_%j.log

source venv/bin/activate
srun python3 train.py


