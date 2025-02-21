#!/bin/bash
#SBATCH --job-name="run_1"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=run_1.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/sgdm_wobn/norm/0.7_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/1.5_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/0.4_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/1.0_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/0.9_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/0.3_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/0.6_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/0.2_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/1.2_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/1.4_1/ --reset
python code/main.py ./data/sgdm_wobn/norm/1.1_1/ --reset
python code/main.py ./data/sgdm_wobn/cnorm/0.8_1/ --reset
python code/main.py ./data/sgdm_wobn/cnorm/1.3_1/ --reset
python code/main.py ./data/sgdm_wobn/cnorm/0.5_1/ --reset
python code/main.py ./data/sgdm_wobn/cnorm/0.7_1/ --reset
python code/main.py ./data/sgdm_wobn/cnorm/1.5_1/ --reset