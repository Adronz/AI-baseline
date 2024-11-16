#!/bin/bash --norc

#SBATCH --account=csb_gpu_acc
#SBATCH --partition=turing
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=45G
#SBATCH --time=07:00:00
#SBATCH --job-name=turing2
#SBATCH --output=sea.log
#SBATCH --error=sea_error.err


source ~/miniconda3/etc/profile.d/conda.sh   # Source the conda script (adjust path if needed)
conda activate kinome  

# Run your Python script
python /dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/breakout.py
