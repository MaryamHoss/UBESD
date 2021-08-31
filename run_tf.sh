#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --account=def-eplourde
#SBATCH --mem 32G
#SBATCH --cpus-per-task 2
#SBATCH --gres=gpu:p100:1
#SBATCH --mail-user=seyedeh.maryam.hosseini.telgerdi@usherbrooke.ca
#SBATCH --mail-type=END

module load python/3.6
source ~/projects/def-eplourde/hoss3301/denv3/bin/activate
cd ~/projects/def-eplourde/hoss3301/work/TrialsOfNeuralVocalRecon
$1