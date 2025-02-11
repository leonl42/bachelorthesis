#!/bin/bash
#SBATCH --job-name="bt"
#SBATCH --cpus-per-task=8             # Number of CPUs per task
#SBATCH --mem=32G                     # Memory per task
#SBATCH --time=20:00:00
# Use GPUs for the job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda/bin/activate deskitopi
python main.py ./exps_adam2/noreg_div8/0.00021/ --reset
python main.py ./exps_adam2/noreg_div8/0.00017/ --reset
python main.py ./exps_adam2/noreg_div8/0.025/ --reset
python main.py ./exps_adam2/noreg_div8/9e-05/ --reset
python main.py ./exps_adam2/noreg_div8/0.00025/ --reset
python main.py ./exps_adam2/noreg_div8/0.00051/ --reset
python main.py ./exps_adam2/noreg_div8/0.001/ --reset
python main.py ./exps_adam2/noreg_div8/0.00069/ --reset
python main.py ./exps_adam2/noreg_div8/0.00625/ --reset
python main.py ./exps_adam2/noreg_div8/0.0016/ --reset
python main.py ./exps_adam2/noreg_div8/0.00039/ --reset
python main.py ./exps_adam2/noreg_div8/0.0001/ --reset
python main.py ./exps_adam2/noreg_div8/0.00015/ --reset
python main.py ./exps_adam2/noreg_div8/0.0004/ --reset
python main.py ./exps_adam2/noreg_div8/0.00278/ --reset
python main.py ./exps_adam2/noreg_div8/0.00013/ --reset
python main.py ./exps_adam2/noreg_div8/0.0028/ --reset
python main.py ./exps_adam2/noreg_div8/0.0005/ --reset
python main.py ./exps_adam2/noreg_div8/0.00011/ --reset
python main.py ./exps_adam2/noreg_div8/0.00031/ --reset
python main.py ./exps_adam2/noreg_div8/8e-05/ --reset
python main.py ./exps_adam2/noreg_div8/0.00156/ --reset
python main.py ./exps_adam2/noreg_div8/0.0007/ --reset
