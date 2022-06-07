for number in {4..17}
do

    echo "$number "
    singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow2.5.0.simg \
            python3 main.py \
            --experiment_id=$number \
            --experiment_name=vision \
            --run=train \
            --gpu_id=0

done