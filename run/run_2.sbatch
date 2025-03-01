#!/bin/bash
#SBATCH --job-name="run_2"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output=run_2.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wbn/cnormu_div4/2.8_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_div4/4.0_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_div4/3.2_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_div4/1.4_1/ --overwrite-num-steps 800000 --overwrite-save-state 800000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/0.4_1/ --overwrite-num-steps 800000 --overwrite-save-state 800000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/2.2_1/ --overwrite-num-steps 500000 --overwrite-save-state 500000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/1.8_1/ --overwrite-num-steps 400000 --overwrite-save-state 400000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/3.4_1/ --overwrite-num-steps 1100000 --overwrite-save-state 1100000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/5.2_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/3.0_1/ --overwrite-num-steps 900000 --overwrite-save-state 900000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/2.4_1/ --overwrite-num-steps 800000 --overwrite-save-state 800000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/4.4_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/2.6_1/ --overwrite-num-steps 600000 --overwrite-save-state 600000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/4.0_1/ --overwrite-num-steps 600000 --overwrite-save-state 600000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/norm/3.2_1/ --overwrite-num-steps 600000 --overwrite-save-state 600000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/2.0_1/ --overwrite-num-steps 600000 --overwrite-save-state 600000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/3.6_1/ --overwrite-num-steps 1000000 --overwrite-save-state 1000000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/0.4_1/ --overwrite-num-steps 400000 --overwrite-save-state 400000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/2.2_1/ --overwrite-num-steps 800000 --overwrite-save-state 800000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/1.8_1/ --overwrite-num-steps 800000 --overwrite-save-state 800000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/3.4_1/ --overwrite-num-steps 1000000 --overwrite-save-state 1000000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/1.2_1/ --overwrite-num-steps 900000 --overwrite-save-state 900000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/5.2_1/ --overwrite-num-steps 1300000 --overwrite-save-state 1300000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnorm_div4/3.0_1/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
