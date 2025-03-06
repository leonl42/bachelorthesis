
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
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && rm -r run && rm run_*.out || echo'
	rsync -a ./run/ hpc:/share/users/student/l/llemke/bachelorthesis/run
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && bash run/run.sh || echo'
	ssh hpc 'squeue -u llemke'

push_run2:
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && rm -r run2 && rm run2_*.out || echo'
	rsync -a ./run2/ hpc:/share/users/student/l/llemke/bachelorthesis/run2
	ssh hpc 'cd /share/users/student/l/llemke/bachelorthesis && bash run2/run.sh || echo'
	ssh hpc 'squeue -u llemke'

squeue:
	ssh hpc 'squeue -u llemke'
 
 
