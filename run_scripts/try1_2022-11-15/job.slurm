#!/bin/sh
#SBATCH --job-name=try1
#SBATCH --open-mode=append
#SBATCH --export=ALL
#SBATCH --time=24:00:00
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
