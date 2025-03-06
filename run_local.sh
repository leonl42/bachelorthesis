
#python code/consecutive_grads.py ./data/adam_wobn/norm/1.4_1/ mg_spacing_0 0 50 2000

#python code/consecutive_grads.py ./data/sgdm_wbn/noreg/0.025/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm_p1/0.044/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm_p1/0.025/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm_p1/0.005/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm/0.15_1/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm/0.2_1/ mg_spacing_0 0 25 4000
#python code/consecutive_grads.py ./data/sgdm_wbn/norm/0.45_1/ mg_spacing_0 0 25 4000
python code/consecutive_grads.py ./data/sgdm_wbn/noreg/0.044/ mg_spacing_0 0 25 4000
python code/consecutive_grads.py ./data/sgdm_wbn/noreg/0.005/ mg_spacing_0 0 25 4000

python code/main.py ./data/sgdm_wbn/noreg/0.099/ --overwrite-num-steps 250000 --overwrite-save-state 250000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/noreg/0.044/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1 
python code/main.py ./data/sgdm_wbn/noreg/0.025/ --overwrite-num-steps 175000 --overwrite-save-state 175000 --overwrite-save-grad -1
#python code/consecutive_grads.py ./data/sgdm_wobn/noreg/0.0128/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/sgdm_wobn/norm/1.0_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/sgdm_wobn/cnorm/0.8_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/sgdm_wobn/cnormu/0.8_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/sgdm_wobn/gcstdu/1.0_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/sgdm_wobn/wd/8e-05/ mg_spacing_0 0 50 2000

#python code/consecutive_grads.py ./data/adam_wobn/noreg/0.0002/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wobn/norm/1.4.1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wobn/cnorm/1.4_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wobn/cnormu/1.2_1/ mg_spacing_0 0 50 2000tmux
#python code/consecutive_grads.py ./data/adam_wobn/gcstdu/1.4_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wobn/wd/8e-05/ mg_spacing_0 0 50 2000

#python code/consecutive_grads.py ./data/adam_wbn/noreg/0.00278/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wbn/norm/4.4_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wbn/cnorm/4.0_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wbn/cnormu/4.4_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wbn/gcstdu/4.0_1/ mg_spacing_0 0 50 2000
#python code/consecutive_grads.py ./data/adam_wbn/wd/3e-08/ mg_spacing_0 0 50 2000
