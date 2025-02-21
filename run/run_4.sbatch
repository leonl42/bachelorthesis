#!/bin/bash
#SBATCH --job-name="run_4"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --output=run_4.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

python code/main.py ./data/sgdm_wobn/gcstdu/0.6_1/ --reset
python code/main.py ./data/sgdm_wobn/gcstdu/0.2_1/ --reset
python code/main.py ./data/sgdm_wobn/gcstdu/1.2_1/ --reset
python code/main.py ./data/sgdm_wobn/gcstdu/1.4_1/ --reset
python code/main.py ./data/sgdm_wobn/gcstdu/1.1_1/ --reset
python code/main.py ./data/adam_wobn/wd/8e-05/ --reset
python code/main.py ./data/adam_wobn/wd/0.00256/ --reset
python code/main.py ./data/adam_wobn/wd/0.00064/ --reset
python code/main.py ./data/adam_wobn/wd/0.00016/ --reset
python code/main.py ./data/adam_wobn/wd/1e-05/ --reset
python code/main.py ./data/adam_wobn/wd/0.00128/ --reset
python code/main.py ./data/adam_wobn/wd/0.00512/ --reset
python code/main.py ./data/adam_wobn/wd/0.00032/ --reset
python code/main.py ./data/adam_wobn/wd/4e-05/ --reset
python code/main.py ./data/adam_wobn/wd/0.01024/ --reset
python code/main.py ./data/adam_wobn/wd/2e-05/ --reset