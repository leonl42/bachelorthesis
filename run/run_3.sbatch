#!/bin/bash
#SBATCH --job-name="run_3"
#SBATCH --cpus-per-task=8            
#SBATCH --mem=32G                     
#SBATCH --time=12:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="H100.80gb"
#SBATCH --output=run_3.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/sgdm_wbn/cnormu_p1/0.005/ --reset
python code/main.py ./data/sgdm_wbn/cnormu_p1/0.006/ --reset
python code/main.py ./data/sgdm_wbn/cnormu_p1/0.008/ --reset
python code/main.py ./data/sgdm_wbn/cnormu_p1/0.011/ --reset