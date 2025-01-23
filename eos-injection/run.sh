#!/bin/bash

#SBATCH --job-name=vi-g
#SBATCH --nodes=1                    # Using 1 node
#SBATCH --gres=gpu:1                 # Using 1 gpu
#SBATCH --time=2-12:00:00            # 1 hour time limit
#SBATCH --ntasks=1
#SBATCH --mem=100000MB                # Using 10GB CPU Memory
#SBATCH --partition=laal_a6000     
#SBATCH --cpus-per-task=4            # Using 4 maximum processor
#SBATCH --output=./logs/%x.%j.out       # Make a log file

source ${HOME}/.bashrc
source ${HOME}/anaconda3/bin/activate
conda activate self-consistency

export MASTER_PORT=12800
master_addr=$(scontrol show hostnames “$SLURM_JOB_NODELIST” | head -n 1)
export MASTER_ADDR=$master_addr

srun python eos-injection.py
srun python greedy.py