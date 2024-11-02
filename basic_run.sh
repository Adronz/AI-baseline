#!/bin/bash --norc

#SBATCH --account=csb_gpu_acc
#SBATCH --partition=turing
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:20:00
#SBATCH --job-name=output
#SBATCH --output=output.log
#SBATCH --error=error.err

source ~/miniconda3/etc/profile.d/conda.sh   # Source the conda script (adjust path if needed)
conda activate kinome  

# Run your Python script
python /dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/VAE_GAN.py
