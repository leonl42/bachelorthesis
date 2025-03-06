#!/bin/bash
#SBATCH --job-name="run_1"
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                     
#SBATCH --time=15:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output=run_1.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/sgdm_wbn/norm_p1/0.099/ --overwrite-num-steps 250000 --overwrite-save-state 250000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/gcstdu_p1/0.099/ --overwrite-num-steps 250000 --overwrite-save-state 250000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/norm_p1/0.044/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/gcstdu_p1/0.044/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/norm_p1/0.025/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/gcstdu_p1/0.025/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_p1/0.00021/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_p1/0.00017/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
python code/main.py ./data/adam_wbn/cnormu_p1/0.00025/ --overwrite-num-steps 1200000 --overwrite-save-state 1200000 --overwrite-save-grad -1 
