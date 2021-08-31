#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --account=def-jrouat
#SBATCH --mem 64G
#SBATCH --cpus-per-task 8
#SBATCH --gres=gpu:0
#SBATCH --mail-user=luca.celotti@usherbrooke.ca
#SBATCH --mail-type=END

module load StdEnv/2020  gcc/9.3.0  cuda/11.0 arrow/1.0.0 python/3.7 scipy-stack
source ~/scratch/denv2/bin/activate
cd ~/scratch/work/TrialsOfNeuralVocalRecon
$1