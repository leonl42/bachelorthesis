#!/bin/bash
#SBATCH --job-name="run_2.1"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=run_2.1.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wbn/norm/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnorm/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnormu/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/gcstdu/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/norm_div4/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnorm_div4/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnormu_div4/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/gcstdu_div4/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/norm_div8/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnorm_div8/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/cnormu_div8/4.0_1/ --reset 
python code/main.py ./data/adam_wbn/gcstdu_div8/4.0_1/ --reset 
