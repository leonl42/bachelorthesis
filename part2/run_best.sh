

python run/wd.py adam 0.00015 --save_model_every 2500 --eval_every 2500 --num_parallel_exps 10
python run/norm.py adam 0.9 10 --save_model_every 2500 --eval_every 2500 --num_parallel_exps 10
python run/center_norm.py adam 0.9 10 --save_model_every 2500 --eval_every 2500 --num_parallel_exps 10