import sys
sys.path.append("../")
from part3.main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('norm_scale', type=float)
parser.add_argument('norm_every', type=int)
parser.add_argument('--run_id',type=str,default="1")
parser.add_argument('--steps', type=int,default=150000)
parser.add_argument('--eval_every', type=int,default=5000)
parser.add_argument('--save_model_every', type=int,default=None)
parser.add_argument('--num_parallel_exps', type=int,default=3)
args = parser.parse_args()


save_path = "./exps_"+args.optim+"/reverse_center_norm_uncenter/" + str(args.norm_scale) + "_" + str(args.norm_every) + "/run_" + args.run_id
train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                    steps=args.steps,
                                    optim=args.optim,
                                    lr=0.0001,
                                    norm_every=args.norm_every,
                                    norm_scale=lambda n,N : args.norm_scale,
                                    norm_fn=weight_reverse_center_normalize_uncenter,
                                    eval_every=args.eval_every,
                                    save_model_every=args.save_model_every))