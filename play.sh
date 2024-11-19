#!/bin/bash --norc
#SBATCH --account=csb_gpu_acc
#SBATCH --partition=turing
#SBATCH --cpus-per-task=6
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=8G
#SBATCH --time=00:02:00
#SBATCH --job-name=turing3
#SBATCH --output=breakout_play.log
#SBATCH --error=breakout_play.err


source ~/miniconda3/etc/profile.d/conda.sh   # Source the conda script (adjust path if needed)
conda activate kinome

# Run your Python script
python /dors/wankowicz_lab/adrian/kinase_colabfold/kinase_new_structs/play_game.py
