#!/bin/bash
#SBATCH --job-name="run"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=11:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=run.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache


python code/main.py ./data/adam_wbn/norm_div8/2.4_1/ --overwrite-num-steps 1000000 --overwrite-save-state 1000000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm_div8/2.6_1/ --overwrite-num-steps 1000000 --overwrite-save-state 1000000 --overwrite-save-grad -1 

python code/main.py ./data/adam_wobn/global_center_std_uncenter/1.5_1/ --overwrite-num-steps 300000 --overwrite-save-state 300000 --overwrite-save-grad -1 


python code/main.py ./data/adam_wbn/wd/5e-07/ --reset 
python code/main.py ./data/adam_wbn/wd/2.5e-07/ --reset 
python code/main.py ./data/adam_wbn/wd/1.2e-07/ --reset 
python code/main.py ./data/adam_wbn/wd/6e-08/ --reset 
python code/main.py ./data/adam_wbn/wd/3e-08/ --reset 
