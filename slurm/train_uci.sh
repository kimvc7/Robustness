#!/bin/bash
#SBATCH --array=0-36
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH -t 10:00:00
#SBATCH -D ./log/
#SBATCH --partition=cbmm
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

cd /om2/user/xboix/src/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

for number in {0..200}
do

    singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om2/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
    python3 main.py \
    --experiment_id=$((37*$number +${SLURM_ARRAY_TASK_ID})) \
    --filesystem=om \
    --experiment_name=uci_all \
    --run=train \
    --gpu_id=0 \
    --missing=True

done
