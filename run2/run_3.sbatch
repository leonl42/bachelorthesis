#!/bin/bash
#SBATCH --job-name="run2_3"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=5:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output=run2_3.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wbn/wd/3e-08/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1