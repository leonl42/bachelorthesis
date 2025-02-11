

py:
	rsync ./main.py hpc:./bachelorthesis/part1/main.py
	rsync ./models.py hpc:./bachelorthesis/part1/models.py
	rsync ./utils.py hpc:./bachelorthesis/part1/utils.py

sbatch:
	rsync -a ./run/ hpc:./bachelorthesis/part1/run

dataset:
	rsync -a ./datasets/ hpc:./bachelorthesis/part1/datasets --info=progress2

settings:
	rsync -a ./exps_adam2/ hpc:./bachelorthesis/part1/exps_adam2 --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --info=progress2

stats:
	rsync -a hpc:./bachelorthesis/part1/exps_adam2/ ./exps_adam2 --info=progress2 --exclude="grads" --exclude="hessians" --exclude="states" 

grads:
	rsync -a hpc:./bachelorthesis/part1/exps_adam2/ ./exps_adam2 --info=progress2 --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats"

states:
	rsync -a hpc:./bachelorthesis/part1/exps_adam2/ ./exps_adam2 --info=progress2 --exclude="grads" --exclude="hessians" --exclude="test_stats" --exclude="train_stats"



