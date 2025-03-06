#!/bin/bash
#SBATCH --job-name="run_10"
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                     
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output=run_10.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wbn/norm_p1/0.00011/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm_p1/0.00031/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wobn/wd/0.00032/ --overwrite-num-steps 900000 --overwrite-save-state 900000 --overwrite-save-grad -1 
