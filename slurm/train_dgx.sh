for number in {0..119}
do

	  
	for gpu in {0..7}
	do

		echo "$number "

		singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow2.4.simg \
			python3 main.py \
			--experiment_id=$((945 + 8*$number + $gpu)) \
			--filesystem=dgx1 \
			--experiment_name=vision \
			--run=train \
			--gpu_id=$gpu &

	done

	wait

done



