from main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('exp_id', type=str)
args = parser.parse_args()

steps = 200000
eval_every = 5000
save_model_every = 25000
num_parallel_exps = 3

# Train model without any regularization
if args.exp_id == "standard":
    for lr in [0.0001,0.00025,0.0005,0.00075,0.001]:
        save_path = "./exps_"+args.optim+"/standard/standard"+str(lr)+"/run_1/"
        train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                            steps=steps,
                                            lr=lr,
                                            optim=args.optim,
                                            eval_every=eval_every,
                                            save_model_every=save_model_every))

# Train model with Weight Decay
if args.exp_id == "wd":
    for wd in [0.0005 + x*0.0005 for x in range(25)]:
        save_path = "./exps_"+args.optim+"/wd/" + str(wd) + "/run_1/"
        train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                            steps=steps,
                                            optim=args.optim,
                                            wd=wd,
                                            lr=0.00025,
                                            eval_every=eval_every,
                                            save_model_every=save_model_every))

# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

# Train Weight Normalization with 0 channel centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "mean_norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/mean_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

if args.exp_id == "mean_std":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/mean_std/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_std,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

# Train Weight Normalization with 0 input centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "reverse_mean_norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/reverse_mean_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_reverse_center_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))
