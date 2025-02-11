#!/bin/bash
#SBATCH --job-name="bt"
#SBATCH --cpus-per-task=8             # Number of CPUs per task
#SBATCH --mem=32G                     # Memory per task
#SBATCH --time=20:00:00
# Use GPUs for the job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda/bin/activate deskitopi

python main.py ./exps_adam2/cnorm_div8/0.8_1/ --reset
python main.py ./exps_adam2/cnorm_div8/2.0_1/ --reset
python main.py ./exps_adam2/cnorm_div8/3.6_1/ --reset
python main.py ./exps_adam2/cnorm_div8/0.4_1/ --reset
python main.py ./exps_adam2/cnorm_div8/2.2_1/ --reset
python main.py ./exps_adam2/cnorm_div8/1.8_1/ --reset
python main.py ./exps_adam2/cnorm_div8/3.4_1/ --reset
python main.py ./exps_adam2/cnorm_div8/1.2_1/ --reset
python main.py ./exps_adam2/cnorm_div8/0.2_1/ --reset
python main.py ./exps_adam2/cnorm_div8/3.0_1/ --reset
python main.py ./exps_adam2/cnorm_div8/2.4_1/ --reset
python main.py ./exps_adam2/cnorm_div8/1.0_1/ --reset
python main.py ./exps_adam2/cnorm_div8/2.6_1/ --reset
python main.py ./exps_adam2/cnorm_div8/2.8_1/ --reset
python main.py ./exps_adam2/cnorm_div8/0.6_1/ --reset
python main.py ./exps_adam2/cnorm_div8/1.6_1/ --reset
python main.py ./exps_adam2/cnorm_div8/3.2_1/ --reset
python main.py ./exps_adam2/cnorm_div8/1.4_1/ --reset
