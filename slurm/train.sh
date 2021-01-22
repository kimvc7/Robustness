#!/bin/bash
#SBATCH --array=306-335
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --exclude=node003,node004,node074,node021,node025,node020,node083,node022,node018,node089,node023
#SBATCH --mem=5GB
#SBATCH -t 2:00:00
#SBATCH -D ./log/
#SBATCH --partition=cbmm
#SBATCH --gres=gpu:1
#SBATCH --constraint=any-gpu

cd /om/user/xboix/src/adversarial/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om/user/xboix/singularity/xboix-tensorflow2.4.simg \
python3 main.py \
--experiment_id=${SLURM_ARRAY_TASK_ID} \
--filesystem=om \
--run=train \
--gpu_id=0




