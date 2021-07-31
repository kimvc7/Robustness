#!/bin/bash
#SBATCH --array=0-999
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH -t 4:00:00
#SBATCH -D ./log/
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

cd /om2/user/xboix/src/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om2/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
python3 main.py \
--experiment_id=$((1890+${SLURM_ARRAY_TASK_ID})) \
--filesystem=om \
--experiment_name=vision \
--run=test_bound \
--gpu_id=0

