#!/bin/bash
#SBATCH --job-name=simple-rl-random-v4
#SBATCH --open-mode=append
#SBATCH --output=/home/gridsan/mdamani/automatic_vehicular_control/slurm_jobs/simple-rl-random-v4/%x_%j.out
#SBATCH --error=/home/gridsan/mdamani/automatic_vehicular_control/slurm_jobs/simple-rl-random-v4/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -N 1
module purge
source ~/.bashrc
module load anaconda/2021b
cd $HOME/automatic_vehicular_control
python full_runner.py --path run_scripts/simple-rl-random-v4_2022-11-21
