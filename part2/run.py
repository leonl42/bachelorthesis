from main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('exp_id', type=str)
parser.add_argument('--run_id',type=str,default="1")
parser.add_argument('--steps', type=int,default=200000)
parser.add_argument('--eval_every', type=int,default=5000)
parser.add_argument('--save_model_every', type=int,default=None)
parser.add_argument('--num_parallel_exps', type=int,default=3)
args = parser.parse_args()


# Train model without any regularization
if args.exp_id == "standard":
    for lr in [0.00025]:

        save_path = "./exps_"+args.optim+"/standard/" + str(lr) + "/run_" + args.run_id
        train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                            steps=args.steps,
                                            lr=lr,
                                            optim=args.optim,
                                            eval_every=args.eval_every,
                                            save_model_every=args.save_model_every))

# Train model with Weight Decay
if args.exp_id == "wd":
    for wd in [0.00005 + x*0.0001 for x in range(10)]:
        save_path = "./exps_"+args.optim+"/wd/" + str(wd) + "/run_" + args.run_id
        train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                            steps=args.steps,
                                            optim=args.optim,
                                            wd=wd,
                                            lr=0.00025,
                                            eval_every=args.eval_every,
                                            save_model_every=args.save_model_every))

# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_normalize,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))

if args.exp_id == "norm_layerwise_stepscale":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/norm_layerwise_stepscale/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                change_scale = lambda n,N,l,L : ((N-n)/N)**((L-l)/L),
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_normalize,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))

# Train Weight Normalization with 0 channel centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "center_norm":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_normalize,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))


if args.exp_id == "center_norm_uncenter":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_norm_uncenter/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_normalize_uncenter,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))

if args.exp_id == "center_std_uncenter":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_std_uncenter/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_std_uncenter,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))

# Train Weight Normalization with 0 input centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "reverse_center_norm":
    for norm_scale in [0.5 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/reverse_center_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_" + args.run_id
            train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                                steps=args.steps,
                                                optim=args.optim,
                                                lr=0.00025,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_reverse_center_normalize,
                                                eval_every=args.eval_every,
                                                save_model_every=args.save_model_every))

