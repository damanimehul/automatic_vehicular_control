#!/bin/sh
#SBATCH --job-name=try1
#SBATCH --open-mode=append
#SBATCH --output=/home/gridsan/mdamani/automatic_vehicular_control/slurm_jobs/try1/%x_%j.out
#SBATCH --error=/home/gridsan/mdamani/automatic_vehicular_control/slurm_jobs/try1/%x_%j.err
#SBATCH --export=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=8G
#SBATCH -c 1
#SBATCH -N 1
module purge
source ~/.bashrc
mdoule load anaconda/2021b
cd $HOME/automatic_vehicular_control
mkdir slurm_jobs/try1
python full_runner.py --path run_scripts/try1_2022-11-16
