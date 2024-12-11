import os
import json
import subprocess
"""
default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 150000, "save_args": {"save_states_every": -1, "save_train_stats_every": 5000, "save_test_stats_every": 5000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "resnet50", "num_classes": 100, "activation_fn": "relu"}, "dataset": {"dataset": "cifar100", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "adam", "lr": 0.0001, "lambda_wd": 0.0015, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}

for norm_fn in ["norm"]:
    for norm_scale in [0.05 + 0.05*x for x in range(10)]:
        for norm_every in [1,10]:
            try:
                os.makedirs("./exps_adam/" + norm_fn + "/" + str(norm_scale) + "_" + str(norm_every),exist_ok=False)
            except:
                continue
            default["optimizer"]["lambda_wd"] = 0
            default["norm"]["change_scale"] = "identity"
            default["norm"]["norm_fn"] = norm_fn
            default["norm"]["norm_multiply"] = norm_scale
            default["norm"]["norm_every"] = norm_every
            with open("./exps_adam/" + norm_fn + "/" + str(norm_scale) + "_" + str(norm_every) + "/settings.json","w") as f:
                json.dump(default,f)

            print("python main.py " + "./exps_adam/" + norm_fn + "/" + str(norm_scale) + "_" + str(norm_every) + "/")



default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 150000, "save_args": {"save_states_every": -1, "save_train_stats_every": 5000, "save_test_stats_every": 5000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "resnet50", "num_classes": 100, "activation_fn": "relu"}, "dataset": {"dataset": "cifar100", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "adam", "lr": 0.0001, "lambda_wd": 0.0015, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}

for wd in [0.0005+0.0005*x for x in range(10)]:
    os.makedirs("./exps_adam/" + "adamw_wd" + "/" + str(wd))
    default["optimizer"]["lambda_wd"] = wd
    with open("./exps_adam/" + "adamw_wd" + "/" + str(wd) + "/settings.json","w") as f:
        json.dump(default,f)

    print("python main.py " + "./exps_adam/" + "adamw_wd" + "/" + str(wd) + "/")
"""

default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 75000, "save_args": {"save_states_every": -1, "save_train_stats_every": 1000, "save_test_stats_every": 1000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "vgg11", "num_classes": 10, "activation_fn": "relu"}, "dataset": {"dataset": "cifar10", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "sgdm", "lr": 0.001, "lambda_wd": 0.0, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}


for norm_fn in ["norm"]:
    for norm_scale in [0.05 + 0.05*x for x in range(10)]:
        for norm_every in [1]:
            try:
                os.makedirs("./exps_sgdm/norm/" + str(norm_scale) + "_" + str(norm_every),exist_ok=False)
            except:
                continue
            default["optimizer"]["lambda_wd"] = 0
            default["norm"]["change_scale"] = "identity"
            default["norm"]["norm_fn"] = norm_fn
            default["norm"]["norm_multiply"] = norm_scale
            default["norm"]["norm_every"] = norm_every
            with open("./exps_sgdm/norm/" + str(norm_scale) + "_" + str(norm_every) + "/settings.json","w") as f:
                json.dump(default,f)

            print("python main.py " + "./exps_sgdm/norm/" + str(norm_scale) + "_" + str(norm_every) + "/")

default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 75000, "save_args": {"save_states_every": -1, "save_train_stats_every": 1000, "save_test_stats_every": 1000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "vgg11", "num_classes": 10, "activation_fn": "relu"}, "dataset": {"dataset": "cifar10", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "sgdm", "lr": 0.001, "lambda_wd": 0.0, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}


for norm_fn in ["norm"]:
    for norm_scale in [0.05 + 0.05*x for x in range(10)]:
        for norm_every in [1]:
            try:
                os.makedirs("./exps_sgdm/norm_slim/" + str(norm_scale) + "_" + str(norm_every),exist_ok=False)
            except:
                continue
            default["model"]["model"] = "vgg11_slim"
            default["optimizer"]["lambda_wd"] = 0
            default["norm"]["change_scale"] = "identity"
            default["norm"]["norm_fn"] = norm_fn
            default["norm"]["norm_multiply"] = norm_scale
            default["norm"]["norm_every"] = norm_every
            with open("./exps_sgdm/norm_slim/" + str(norm_scale) + "_" + str(norm_every) + "/settings.json","w") as f:
                json.dump(default,f)

            print("python main.py " + "./exps_sgdm/norm_slim/" + str(norm_scale) + "_" + str(norm_every) + "/")

default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 75000, "save_args": {"save_states_every": -1, "save_train_stats_every": 1000, "save_test_stats_every": 1000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "vgg11", "num_classes": 10, "activation_fn": "relu"}, "dataset": {"dataset": "cifar10", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "sgdm", "lr": 0.001, "lambda_wd": 0.0, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}

for wd in [0.001+0.001*x for x in range(10)]:
    os.makedirs("./exps_sgdm/" + "wd" + "/" + str(wd))
    default["optimizer"]["lambda_wd"] = wd
    with open("./exps_sgdm/" + "wd" + "/" + str(wd) + "/settings.json","w") as f:
        json.dump(default,f)

    print("python main.py " + "./exps_sgdm/" + "wd" + "/" + str(wd) + "/")

default = {"num_devices": 1, "num_experiments_per_device": 3, "random_key": 42, "num_steps": 75000, "save_args": {"save_states_every": -1, "save_train_stats_every": 1000, "save_test_stats_every": 1000, "save_grad_every": -1, "save_hessian_every": -1}, "model": {"model": "vgg11", "num_classes": 10, "activation_fn": "relu"}, "dataset": {"dataset": "cifar10", "batch_size": 128, "dataset_path": "./datasets/"}, "optimizer": {"optimizer": "sgdm", "lr": 0.001, "lambda_wd": 0.0, "momentum": 0.9, "apply_wd_every": 1}, "norm": {"change_scale" : "identity","norm_fn": "identity", "norm_multiply": 1, "norm_every": -1, "reverse_norms": False}, "at_step": 0}

for wd in [0.001+0.001*x for x in range(10)]:
    os.makedirs("./exps_sgdm/" + "wd_slim" + "/" + str(wd))
    default["optimizer"]["lambda_wd"] = wd
    default["model"]["model"] = "vgg11_slim"
    with open("./exps_sgdm/" + "wd_slim" + "/" + str(wd) + "/settings.json","w") as f:
        json.dump(default,f)

    print("python main.py " + "./exps_sgdm/" + "wd_slim" + "/" + str(wd) + "/")