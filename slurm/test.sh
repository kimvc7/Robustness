#!/bin/bash
#SBATCH --array=63
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=4GB
#SBATCH -t 0:05:00
#SBATCH -D ./log/
#SBATCH --partition=use-everything
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

cd /om/user/xboix/src/adversarial/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
python3 main.py \
--experiment_id=$((0+${SLURM_ARRAY_TASK_ID})) \
--filesystem=om \
--experiment_name=fashion \
--run=test \
--gpu_id=0




