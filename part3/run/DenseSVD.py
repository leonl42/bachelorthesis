import sys
sys.path.append("../")
from part2.main import *
from part3.main_optsvd import train as train_optsvd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('loss_svd_scale', type=float)
parser.add_argument('--run_id',type=str,default="1")
parser.add_argument('--steps', type=int,default=150000)
parser.add_argument('--eval_every', type=int,default=5000)
parser.add_argument('--save_model_every', type=int,default=None)
parser.add_argument('--num_parallel_exps', type=int,default=3)
args = parser.parse_args()

save_path = "./test_" + args.optim + "/DenseSVD/"+str(args.loss_svd_scale)+"/run_" + args.run_id
train_optsvd(save_path, SimpleNamespaceNone(num_parallel_exps=args.num_parallel_exps,
                                            steps=args.steps,
                                            lr=0.0001,
                                            optim=args.optim,
                                            eval_every=args.eval_every,
                                            save_dense_every=args.save_model_every,
                                            loss_svd_scale = args.loss_svd_scale))