from main import *
import argparse
import jax.numpy as jnp

parser = argparse.ArgumentParser()
parser.add_argument('optim', type=str)
parser.add_argument('exp_id', type=str)
args = parser.parse_args()

steps = 150000
eval_every = 5000
save_model_every = 25000
num_parallel_exps = 3

# Train model without any regularization
if args.exp_id == "standard":
    for lr in [0.00005,0.0001,0.00015]:
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
                                            lr=0.0001,
                                            eval_every=eval_every,
                                            save_model_every=save_model_every))
        
# Train model with Singular Value Regularization
if args.exp_id == "svd_scale":
    for shift in [0.9,0.5,0.1]:
        for scale in [0.001,0.0015,0.01]:
            for scale_every in [100]:
                save_path = "./exps_"+args.optim+"/svd_scale/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_1/"
                train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                    steps=steps,
                                                    optim=args.optim,
                                                    lr=0.0001,
                                                    eval_every=eval_every,
                                                    svd_p = lambda n,N : (shift,scale),
                                                    svd_fn=svd_scale,
                                                    svd_scale_every=scale_every,
                                                    save_model_every=save_model_every))

# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.0001,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

            
# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm_stepscale":
    save_path = "./exps_"+args.optim+"/norm_stepscale/0.5_100/run_1/"
    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                        steps=steps,
                                        optim=args.optim,
                                        lr=0.0001,
                                        norm_every=100,
                                        change_scale = lambda n,N,l,L : (N-n)/N,
                                        norm_scale=lambda n,N : 0.5,
                                        norm_fn=weight_normalize,
                                        eval_every=eval_every,
                                        save_model_every=save_model_every))

# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm_cutoff":
    save_path = "./exps_"+args.optim+"/norm_cutoff/0.5_100/run_1/"
    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                        steps=steps,
                                        optim=args.optim,
                                        lr=0.0001,
                                        norm_every=100,
                                        change_scale = lambda n,N,l,L : jnp.heaviside((N-n)/N-0.5,0),
                                        norm_scale=lambda n,N : 0.5,
                                        norm_fn=weight_normalize,
                                        eval_every=eval_every,
                                        save_model_every=save_model_every))

# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm_cutoff_reverse":
    save_path = "./exps_"+args.optim+"/norm_cutoff_reverse/0.5_100/run_1/"
    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                        steps=steps,
                                        optim=args.optim,
                                        lr=0.0001,
                                        norm_every=100,
                                        change_scale = lambda n,N,l,L : jnp.heaviside((0.5-(N-n)/N),1),
                                        norm_scale=lambda n,N : 0.5,
                                        norm_fn=weight_normalize,
                                        eval_every=eval_every,
                                        save_model_every=save_model_every))
    
# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm_layerwise_stepscale":
    save_path = "./exps_"+args.optim+"/norm_layerwise_stepscale/0.5_100/run_1/"
    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                        steps=steps,
                                        optim=args.optim,
                                        lr=0.0001,
                                        norm_every=100,
                                        change_scale = lambda n,N,l,L : ((N-n)/N)**((L-l)/L),
                                        norm_scale=lambda n,N : 0.5,
                                        norm_fn=weight_normalize,
                                        eval_every=eval_every,
                                        save_model_every=save_model_every))
    
# Train Weight Normalization (w = c/||w||)
if args.exp_id == "norm_stepscale_reverse":
    save_path = "./exps_"+args.optim+"/norm_stepscale_reverse/0.5_100/run_1/"
    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                        steps=steps,
                                        optim=args.optim,
                                        lr=0.0001,
                                        norm_every=100,
                                        change_scale = lambda n,N,l,L : 1-((N-n)/N),
                                        norm_scale=lambda n,N : 0.5,
                                        norm_fn=weight_normalize,
                                        eval_every=eval_every,
                                        save_model_every=save_model_every))

# Train Weight Normalization with 0 channel centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "center_norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.0001,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

if args.exp_id == "center_norm_uncenter":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_norm_uncenter/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.0001,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_normalize_uncenter,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

if args.exp_id == "center_std_uncenter":
    for norm_scale in [0.1 + x*0.1 for x in range(20)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/center_std_uncenter/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.0001,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_center_std_uncenter,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

# Train Weight Normalization with 0 input centering (w = (w-mean(w))/||w-mean(w)||)
if args.exp_id == "reverse_center_norm":
    for norm_scale in [0.1 + x*0.1 for x in range(10)]:
        for norm_every in [1,10,100]:
            save_path = "./exps_"+args.optim+"/reverse_center_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/"
            train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                steps=steps,
                                                optim=args.optim,
                                                lr=0.0001,
                                                norm_every=norm_every,
                                                norm_scale=lambda n,N : norm_scale,
                                                norm_fn=weight_reverse_center_normalize,
                                                eval_every=eval_every,
                                                save_model_every=save_model_every))

if args.exp_id == "svd_smoothing":
    for smoothing_factor in [(12.0,6.0)]:
        for scale in [0.95,0.98,0.99]:
            for scale_every in [50]:
                save_path = "./exps_"+args.optim+"/svd_smoothing/" + str(smoothing_factor) + "_" + str(scale) + "_" + str(scale_every) + "/run_1/"
                train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                    steps=steps,
                                                    optim=args.optim,
                                                    lr=0.0001,
                                                    eval_every=eval_every,
                                                    svd_p = lambda n,N : (scale,(((N-n)/N)**2)*smoothing_factor[0]+ (1-(((N-n)/N)**2))*smoothing_factor[1]),
                                                    svd_fn = svd_smoothing,
                                                    svd_scale_every=scale_every,
                                                    save_model_every=save_model_every))

if args.exp_id == "svd_static_exp_fit":
    for scale_every in [100,50,10]:
        for a in [0,0.5,1]:
            for b in [0.5,1.5,1.5,5,10,15]:
                for c in [0,0.1,0.5,1,1.5]:
                    save_path = "./exps_"+args.optim+"/svd_static_exp_fit/" + str(a) + "_" + str(b) + "_" + str(c) + "_" + str(scale_every) + "/run_1/"
                    train(save_path, SimpleNamespaceNone(num_parallel_exps=num_parallel_exps,
                                                        steps=steps,
                                                        optim=args.optim,
                                                        lr=0.0001,
                                                        eval_every=eval_every,
                                                        svd_p = lambda n,N : (a,b,c),
                                                        svd_fn = svd_static_exp_fit,
                                                        svd_scale_every=scale_every,
                                                        save_model_every=save_model_every))




#train("./exps/best/wd/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,wd_dense=0.006,lr=0.001,eval_every=1335,save_dense_every=1335))
#train("./exps/best/norm/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=1,norm_scale=lambda n,N : 0.2,norm_fn=weight_normalize,eval_every=1335,save_dense_every=1335))
#train("./exps/best/norm/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=1,norm_scale=lambda n,N : 0.2,norm_fn=weight_normalize,eval_every=1335,save_dense_every=1335))
#train("./exps/best/mean_norm/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=25,norm_scale=lambda n,N : 0.2,norm_fn=weight_normalize,eval_every=1335,save_dense_every=1335))
#train("./exps/best/svd_scale/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=1335,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.015,svd_fn=svd_scale,svd_scale_every=50,save_dense_every=1335))
#train("./exps/best/svd_exp_fit/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=1335,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 2,svd_fn = svd_exp_fit,svd_scale_every=100,save_dense_every=1335))
#train("./exps/best/svd_static_exp_fit/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=1335,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.01,svd_fn = svd_static_exp_fit,svd_scale_every=50,save_dense_every=1335))
