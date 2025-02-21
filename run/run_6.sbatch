#!/bin/bash
#SBATCH --job-name="run_6"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=run_6.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/adam_wobn/cnorm/1.0_1/ --reset
python code/main.py ./data/adam_wobn/cnorm/0.9_1/ --reset
python code/main.py ./data/adam_wobn/cnorm/0.6_1/ --reset
python code/main.py ./data/adam_wobn/cnorm/1.2_1/ --reset
python code/main.py ./data/adam_wobn/cnorm/1.4_1/ --reset
python code/main.py ./data/adam_wobn/cnorm/1.1_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/0.8_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/1.3_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/0.5_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/0.7_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/1.5_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/1.0_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/0.9_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/0.6_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/1.2_1/ --reset
python code/main.py ./data/adam_wobn/cnormu/1.4_1/ --reset