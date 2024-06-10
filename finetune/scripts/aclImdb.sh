#!/bin/bash


#SBATCH --job-name=aclImdb

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

#SBATCH --mem=128G
#SBATCH --time=1-00:00:00

#SBATCH --mail-user=<dphuong@iastate.edu>
#SBATCH --mail-type=END

#SBATCH --output=aclImdb.out
#SBATCH --error=aclImdb.err

#SBATCH --exclude=singularity


echo "Loading modules"

#module load ml-gpu

module load miniconda3/22.11.1-hydt3qz

source activate flora_pronto

#source /work/LAS/jannesar-lab/dphuong/.conda/bin/activate /work/LAS/jannesar-lab/dphuong/.conda/envs/flora_pronto

cd /work/LAS/jannesar-lab/dphuong/fedtext/finetune

nvidia-smi -L
nvidia-smi --query-gpu=compute_cap --format=csv

echo "$PWD"

echo "Started $SLURM_JOB_NAME at $(date)"


time python gridsearch.py \
    -data aclImdb \
    -sd 0


echo "Finish $SLURM_JOB_NAME at $(date)"