#!/bin/sh
#SBATCH --job-name=try1
#SBATCH --open-mode=append
#SBATCH --output=$HOME/automatic_vehicular_control/slurm_jobs/test-1/%a_%x_%j.out
#SBATCH --error=$HOME/automatic_vehicular_control/slurm_jobs/test-1/%a_%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:0
#SBATCH --mem=8G
#SBATCH --mail-type=END
#SBATCH --mail-user=mehul42@mit.edu
#SBATCH -c 1
#SBATCH -N 1
module purge
source ~/.bashrc
mdoule load anaconda/2021b
cd $HOME/automatic_vehicular_control
python full_runner.py --path run_scripts/try1_2022-11-15
