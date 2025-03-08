#!/bin/bash
#SBATCH --job-name="run_2"
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                     
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="H100.80gb"
#SBATCH --output=run_2.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wbn/cnorm_p1_lrfix/0.00069/ --reset
python code/main.py ./data/adam_wbn/cnorm_p1_lrfix/0.00025/ --reset
python code/main.py ./data/adam_wbn/cnorm_p1_lrfix/0.00011/ --reset
