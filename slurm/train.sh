#!/bin/bash
#SBATCH --array=675-855
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --exclude=node028,node022,node030,node025,node023,node004,node003,node021,node022,node026,node089
#SBATCH --mem=20GB
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
--experiment_id=$((0+${SLURM_ARRAY_TASK_ID})) \
--filesystem=om \
--experiment_name=synthetic \
--run=train \
--gpu_id=0




