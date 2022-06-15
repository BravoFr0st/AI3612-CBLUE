#!/bin/bash
#SBATCH --job-name=run_cmeee
#SBATCH -p 64c512g
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --output=../logs/run_cmeee-%A.log
#SBATCH --error=../logs/run_cmeee-%A.log

python ee_data.py