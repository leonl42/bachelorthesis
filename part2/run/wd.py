import sys
sys.path.append("../")
from part2.main import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('wd', type=float)
parser.add_argument('--model', type=str,default='vgg11')
parser.add_argument('--run_id',type=str,default="1")
parser.add_argument('--steps', type=int,default=200000)
parser.add_argument('--eval_every', type=int,default=5000)
parser.add_argument('--save_model_every', type=int,default=None)
parser.add_argument('--num_parallel_exps', type=int,default=3)
args = parser.parse_args()

if args.model == "vgg11":
    save_path = "./exps_"+args.optim+"/wd/" + str(args.wd) + "/run_" + args.run_id
else:
    save_path = "./exps_"+args.optim+"/wd_" + args.model + "/" + str(args.wd) + "/run_" + args.run_id

train(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                    steps=args.steps,
                                    model=args.model,
                                    optim=args.optim,
                                    wd=args.wd,
                                    lr=0.00025,
                                    eval_every=args.eval_every,
                                    save_model_every=args.save_model_every))