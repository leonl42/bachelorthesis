
#python run.py ./exps_adam/cd_0.25_1/ 
#python run.py ./exps_adam/cd_1_1/ 
#python run.py ./exps_adam/no_wd/ 
python run.py ./exps_adam/no_wd/ --reset
python run.py ./exps_adam/wd0.005/ --reset
python run.py ./exps_adam/wd0.05_lr0.001/ --reset
python run.py ./exps_adam/wd0.0005_lr0.001/ --reset
python run.py ./exps_adam/wd0.005_lr0.0001/ --reset
python run.py ./exps_adam/wd0.0005_lr0.0001/ --reset
python run.py ./exps_adam/wd0.005/ --reset