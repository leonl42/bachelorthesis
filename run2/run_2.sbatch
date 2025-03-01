#!/bin/bash
#SBATCH --job-name="swobn_2"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output=swobn_2.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/sgdm_wobn/cnorm/0.8_1/ --overwrite-num-steps 800000 --overwrite-save-state 25000 --overwrite-save-grad 25000 --reset
python code/main.py ./data/sgdm_wobn/gcstdu/1.0_1/ --overwrite-save-state 25000 --overwrite-save-grad 25000 --reset 