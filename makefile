

py:
	rsync ./main.py hpc:./bachelorthesis/part1/main.py
	rsync ./models.py hpc:./bachelorthesis/part1/models.py
	rsync ./utils.py hpc:./bachelorthesis/part1/utils.py


dataset:
	rsync -a ./datasets/ hpc:./bachelorthesis/part1/datasets --info=progress2

settings_adam_wbn:
	rsync -a ./data/adam_wbn hpc:./bachelorthesis/data/adam_wbn --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --info=progress2

pull_stats:
	rsync -a hpc:./bachelorthesis/data/adam_wbn/ ./data/adam_wbn --info=progress2 --exclude="grads" --exclude="hessians" --exclude="states" --exclude="datasets"
	rsync -a hpc:./bachelorthesis/data/adam_wobn/ ./data/adam_wobn --info=progress2 --exclude="grads" --exclude="hessians" --exclude="states" --exclude="datasets"
	rsync -a hpc:./bachelorthesis/data/sgdm_wobn/ ./data/sgdm_wobn --info=progress2 --exclude="grads" --exclude="hessians" --exclude="states" --exclude="datasets"

push_settings:
	rsync -a ./data/ hpc:./bachelorthesis/data --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --exclude="datasets" --exclude="sgdm_wbn" --info=progress2

pull_settings:
	rsync -a hpc:./bachelorthesis/data/ ./data --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --exclude="datasets" --exclude="sgdm_wbn" --info=progress2

push_run:
	rsync -a ./run/ hpc:./bachelorthesis/run
