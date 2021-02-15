#!/bin/bash
#SBATCH --array=0,7,9,37,39,42,82,97,172,173,181,232,252,297,341,406,427,428,438,535,575,606,613,641,643,674,707,710,733,746,773,778,825,844,884,891,910,946,985,998
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --exclude=node077,node071,node072,node073,node074,node075,node076,node028,node022,node030,node025,node023,node004,node003,node021,node026,node022,node089
#SBATCH --mem=4GB
#SBATCH -t 0:5:00
#SBATCH -D ./log/
#SBATCH --partition=use-everything
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
--experiment_name=uci \
--run=test \
--gpu_id=0




