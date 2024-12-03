import os
import json
import subprocess

default = {"num_devices": 1, "num_experiments_per_device": 2, "random_key": 42, "num_steps": 150000, "save_args": {"save_states_every": -1, "save_train_stats_every": 5000, "save_test_stats_every": 5000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "resnet50", "num_classes": 100, "activation_fn": "relu"}, "dataset": {"dataset": "cifar100", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "adam", "lr": 0.001, "lambda_wd": 0.005, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}

for lr in [0.0001,0.0005,0.001,0.0015]:

    default["optimizer"]["lr"] = lr
    default["optimizer"]["lambda_wd"] = 0

    os.makedirs("./exps_adam/no_wd_lr" + str(lr))
    with open("./exps_adam/no_wd_lr" + str(lr) + "/settings.json","w") as f:
        json.dump(default,f)
    for wd in [0.0001,0.0005,0.001,0.0015]:
        default["optimizer"]["lr"] = lr
        default["optimizer"]["lambda_wd"] = wd
        os.makedirs("./exps_adam/wd" + str(wd) + "_lr" + str(lr))
        with open("./exps_adam/wd" + str(wd) + "_lr" + str(lr)+ "/settings.json","w") as f:
            json.dump(default,f)

