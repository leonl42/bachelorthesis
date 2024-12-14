
python run/global_center_std_uncenter.py sgdm 1.2 1 --lr 0.025 --suffix _lr0.025
python run/global_center_std_uncenter.py sgdm 1.2 1 --lr 0.0075 --suffix _lr0.0075
python run/global_center_std_uncenter.py sgdm 1.2 1 --lr 0.0025 --suffix _lr0.0025
python run/global_center_std_uncenter.py sgdm 1.2 1 --lr 0.00075 --suffix _lr0.00075
python run/global_center_std_uncenter.py sgdm 1.2 1 --lr 0.00025 --suffix _lr0.00025

python run/norm.py sgdm 0.9 10 --lr 0.025 --suffix _lr0.025
python run/norm.py sgdm 0.9 10 --lr 0.0075 --suffix _lr0.0075
python run/norm.py sgdm 0.9 10 --lr 0.0025 --suffix _lr0.0025
python run/norm.py sgdm 0.9 10 --lr 0.00075 --suffix _lr0.00075
python run/norm.py sgdm 0.9 10 --lr 0.00025 --suffix _lr0.00025

python run/reverse_center_norm.py sgdm 0.5 1 --lr 0.075
python run/reverse_center_norm.py sgdm 0.6 1 --lr 0.075
python run/reverse_center_norm.py sgdm 0.7 1 --lr 0.075
python run/reverse_center_norm.py sgdm 0.8 1 --lr 0.075
python run/reverse_center_norm.py sgdm 0.9 1 --lr 0.075
python run/reverse_center_norm.py sgdm 1.0 1 --lr 0.075
python run/reverse_center_norm.py sgdm 1.1 1 --lr 0.075
python run/reverse_center_norm.py sgdm 1.2 1 --lr 0.075
python run/reverse_center_norm.py sgdm 1.3 1 --lr 0.075
python run/reverse_center_norm.py sgdm 1.4 1 --lr 0.075
