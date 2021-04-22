for number in {0..9}
do

	  
	for gpu in {0..7}
	do

		echo "$number "

		singularity exec --nv /raid/poggio/home/xboix/containers/xboix-tensorflow2.4.simg \
			python3 main.py \
			--experiment_id=$((238 + 8*$number + $gpu)) \
			--filesystem=dgx1 \
			--experiment_name=threelayer \
			--run=test \
			--gpu_id=$gpu &

	done

	wait

done



