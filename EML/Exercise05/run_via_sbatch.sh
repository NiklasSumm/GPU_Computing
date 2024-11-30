#!/bin/bash
#SBATCH --gres gpu:1
#SBATCH --mem 10G
#SBATCH --cpus-per-task 2
#SBATCH --time 00:30:00
#SBATCH -p exercise
#SBATCH -o slurm_output.log

echo "Loading conda env"
eval "$(command conda 'shell.bash' 'hook' 2> /dev/null)"
conda activate eml

echo "Running exercise05_template.py"
python exercise05_template.py
