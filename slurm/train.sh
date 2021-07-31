#!/bin/bash
#SBATCH --array=0-39
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH -t 24:00:00
#SBATCH -D ./log/
#SBATCH --exclude=node093,node094,node021,node028
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu


cd /om2/user/xboix/src/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

for number in {0..117}
do

    singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om2/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
    python3 main.py \
    --experiment_id=$((40*$number +${SLURM_ARRAY_TASK_ID})) \
    --filesystem=om \
    --experiment_name=vision_small \
    --run=train \
    --gpu_id=0 \
    --missing=True

done


