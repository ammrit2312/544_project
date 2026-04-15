#!/bin/bash
#SBATCH --account=robinjia_1822
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --job-name=gec_train
#SBATCH --output=train_%j.log

nvidia-smi
source /home1/aamrit/src/venv/bin/activate
cd /home1/aamrit/src
srun python3 train.py
