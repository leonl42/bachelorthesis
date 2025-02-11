#!/bin/bash
#SBATCH --job-name="bt"
#SBATCH --cpus-per-task=32            # Number of CPUs per task
#SBATCH --mem=64G                     # Memory per task
#SBATCH --time=30:00:00
# Use GPUs for the job
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

source ~/miniconda/bin/activate deskitopi

python main.py ./exps_adam2/gcstdu_div4/0.8_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/2.0_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/3.6_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/0.4_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/2.2_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/1.8_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/3.4_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/1.2_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/0.2_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/3.0_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/2.4_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/1.0_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/2.6_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/2.8_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/0.6_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/1.6_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/3.2_1/ --reset
python main.py ./exps_adam2/gcstdu_div4/1.4_1/ --reset

python main.py ./exps_adam2/noreg_div4/0.00021/ --reset
python main.py ./exps_adam2/noreg_div4/0.00017/ --reset
python main.py ./exps_adam2/noreg_div4/0.025/ --reset
python main.py ./exps_adam2/noreg_div4/9e-05/ --reset
python main.py ./exps_adam2/noreg_div4/0.00025/ --reset
python main.py ./exps_adam2/noreg_div4/0.00051/ --reset
python main.py ./exps_adam2/noreg_div4/0.001/ --reset
python main.py ./exps_adam2/noreg_div4/0.00069/ --reset
python main.py ./exps_adam2/noreg_div4/0.00625/ --reset
python main.py ./exps_adam2/noreg_div4/0.0016/ --reset
python main.py ./exps_adam2/noreg_div4/0.00039/ --reset
python main.py ./exps_adam2/noreg_div4/0.0001/ --reset
python main.py ./exps_adam2/noreg_div4/0.00015/ --reset
python main.py ./exps_adam2/noreg_div4/0.0004/ --reset
python main.py ./exps_adam2/noreg_div4/0.00278/ --reset
python main.py ./exps_adam2/noreg_div4/0.00013/ --reset
python main.py ./exps_adam2/noreg_div4/0.0028/ --reset
python main.py ./exps_adam2/noreg_div4/0.0005/ --reset
python main.py ./exps_adam2/noreg_div4/0.00011/ --reset
python main.py ./exps_adam2/noreg_div4/0.00031/ --reset
python main.py ./exps_adam2/noreg_div4/8e-05/ --reset
python main.py ./exps_adam2/noreg_div4/0.00156/ --reset
python main.py ./exps_adam2/noreg_div4/0.0007/ --reset

python main.py ./exps_adam2/wd_div4/0.002048/ --reset
python main.py ./exps_adam2/wd_div4/3.2e-05/ --reset
python main.py ./exps_adam2/wd_div4/2e-06/ --reset
python main.py ./exps_adam2/wd_div4/0.000256/ --reset
python main.py ./exps_adam2/wd_div4/0.032768/ --reset
python main.py ./exps_adam2/wd_div4/4e-06/ --reset
python main.py ./exps_adam2/wd_div4/0.016384/ --reset
python main.py ./exps_adam2/wd_div4/1.6e-05/ --reset
python main.py ./exps_adam2/wd_div4/0.001024/ --reset
