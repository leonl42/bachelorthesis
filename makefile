
push_code:
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && rm -r code || echo'
	rsync -r ./code/ hpc:/share/users/student/l/llemke/bachelorthesis/code

pull_stats:
	rsync -a hpc:/share/users/student/l/llemke/bachelorthesis/data/ ./data --info=progress2 --exclude="grads" --exclude="hessians" --exclude="states" --exclude="datasets"
	
push_settings:
	rsync -a ./data/ hpc:/share/users/student/l/llemke/bachelorthesis/data --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --exclude="datasets" --exclude="sgdm_wbn" --info=progress2

pull_settings:
	rsync -a hpc:/share/users/student/l/llemke/bachelorthesis/data/ ./data --exclude="grads" --exclude="hessians" --exclude="states" --exclude="test_stats" --exclude="train_stats" --exclude="datasets" --exclude="sgdm_wbn" --info=progress2

push_run:
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && rm -r run && rm *.out || echo'
	rsync -a ./run/ hpc:/share/users/student/l/llemke/bachelorthesis/run
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && bash run/run.sh || echo'

squeue:
	ssh hpc 'squeue -u llemke'