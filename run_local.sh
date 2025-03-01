#python code/consecutive_grads.py ./data/sgdm_wobn/cnorm/0.8_1/ mg_spacing_0 0 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/noreg/0.0128/ mg_spacing_0 0 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/norm/1.0_1/ mg_spacing_0 0 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/cnormu/0.8_1/ mg_spacing_0 0 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/cnorm/0.8_1/ mg_spacing_175000 175000 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/cnormu/0.8_1/ mg_spacing_0 0 50 500
#python code/consecutive_grads.py ./data/sgdm_wobn/cnormu/0.8_1/ mg_spacing_175000 175000 50 500

#python code/consecutive_grads.py ./data/sgdm_wobn/noreg/0.0128/ mg_spacing_0 0 100 1000
#python code/consecutive_grads.py ./data/sgdm_wobn/norm/1.0_1/ mg_spacing_0 0 100 1000
#python code/consecutive_grads.py ./data/sgdm_wobn/cnorm/0.8_1/ mg_spacing_0 0 100 1000
#python code/consecutive_grads.py ./data/sgdm_wobn/cnormu/0.8_1/ mg_spacing_0 0 100 1000
#python code/consecutive_grads.py ./data/sgdm_wobn/gcstdu/1.0_1/ mg_spacing_0 0 100 1000

python code/main.py ./data/adam_wobn/noreg/0.0002/ --reset
python code/main.py ./data/adam_wobn/noreg/0.0004/ --reset

python code/consecutive_grads.py ./data/sgdm_wobn/wd/8e-05/ mg_spacing_0 0 100 1000
python code/consecutive_grads.py ./data/adam_wobn/noreg/0.0002/ mg_spacing_0 0 100 1000
python code/consecutive_grads.py ./data/adam_wobn/norm/1.4_1/ mg_spacing_0 0 100 1000
python code/consecutive_grads.py ./data/adam_wobn/cnorm/1.4_1/ mg_spacing_0 0 100 1000
python code/consecutive_grads.py ./data/adam_wobn/cnormu/1.2_1/ mg_spacing_0 0 100 1000