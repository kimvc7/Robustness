#!/bin/bash
#SBATCH --array=568,574,575,576,577,578,580,583,585,586,588,589,591,593,599,603,605,606,608,612,615,618,619,620,623,626,868,869,875,879,896,926,935
#SBATCH -n 1
#SBATCH -c 2
#SBATCH --job-name=robustness
#SBATCH --mem=8GB
#SBATCH -t 4:00:00
#SBATCH -D ./log/
#SBATCH --partition=normal
#SBATCH --gres=gpu:1
#SBATCH --constraint=high-capacity

cd /om2/user/xboix/src/Robustness/

hostname
echo $CUDA_VISIBLE_DEVICES
echo $CUDA_DEVICE_ORDER

singularity exec -B /om:/om -B /om2:/om2 -B /scratch/user/xboix:/vast --nv /om2/user/xboix/singularity/xboix-tensorflow2.5.0.simg \
python3 main.py \
--experiment_id=$((0+${SLURM_ARRAY_TASK_ID})) \
--filesystem=om \
--experiment_name=threelayer \
--run=train \
--gpu_id=0


