import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
#import sys
#sys.stderr = open(os.devnull, "w") 
import jax
import jax.numpy as jnp
# Perform a Jax operation before importing Tensorflow to ensure Jax initializes Cuda before Tensorflow.
a = jnp.ones((4,4))@(2*jnp.ones((4,4)))
import argparse
import pickle as pkl
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental import mesh_utils
import tensorflow as tf 
tf.config.set_visible_devices([], 'GPU')
from utils import *
import json
from tqdm import tqdm
import time
import os
import shutil



parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str)
parser.add_argument('load_from_mg_spacing', type=str)
parser.add_argument('--bfloat16', action=argparse.BooleanOptionalAction)
parse_args = parser.parse_args()

if os.path.exists(os.path.join(parse_args.save_path,"adam_drift")):
    shutil.rmtree(os.path.join(parse_args.save_path,"adam_drift"))


os.makedirs(os.path.join(parse_args.save_path,"adam_drift"),exist_ok=True)

with open(parse_args.save_path + "settings.json", "r") as f:
    args_dict = json.load(f)

args = dict_to_namespace(args_dict)

# Get the cpu device in order to put all params on the cpu before distributing them to the other devices (either cpu's or gpu's)
default_cpu_device = jax.devices("cpu")[0]

# Create device mesh in order to distribute params on devices
devices = mesh_utils.create_device_mesh((args.num_devices,))
mesh = Mesh(devices, axis_names=('d',))
named_sharding = NamedSharding(mesh, P('d'))

# Initialize the random seed
split_key = jax.random.key(args.random_key)

ds_train,ds_test = get_dataset(args,parse_args.bfloat16)

model,layer_depth_dict,num_layers = get_model(args)

# Get key paths of all layers
with jax.default_device(default_cpu_device):
    helper_weights = model.init(jax.random.key(0),jnp.ones((1,32,32,3)))["params"]

optimizer = get_optimizer(args,helper_weights=helper_weights)

with open(args.save_path + "states/" + str(0) + ".pkl", "rb") as f:
    init_weights,_,_ = pkl.load(f)
    init_weights = device_put(named_sharding,init_weights)[0]


for ckpt in tqdm(list(os.listdir(os.path.join(args.save_path,parse_args.load_from_mg_spacing,"states")))):
    with open(os.path.join(args.save_path,parse_args.load_from_mg_spacing,"states",str(ckpt)), "rb") as f:
        (weights,batch_stats,optimizer_state) = pkl.load(f)

    weights,batch_stats,optimizer_state = device_put(named_sharding,weights,batch_stats,optimizer_state)

    if args.norm.norm_fn == "center_std_uncenter" or args.norm.norm_fn == "global_center_std_uncenter":
        with open(args.save_path + "states/" + str(0) + ".pkl", "rb") as f:
            std_weights,_,_ = pkl.load(f)
            std_weights = device_put(named_sharding,std_weights)[0]


    if not args.norm.apply_norm_to:
        apply_norm_to = "conv&kernel"
    else:
        apply_norm_to = args.norm.apply_norm_to

    # If we want to use the normalization scheme proposed by Niehaus et al. 2024, we have to calculate the standard deviation before training
    if args.norm.norm_fn == "center_std_uncenter":
        # Get the standard deviations of the weights in the beginning
        target_std = tree_map_with_path(lambda s,w : jax.vmap(lambda x : jnp.std(x,axis=tuple(range(len(x.shape)-1)),keepdims=True),in_axes=(0,))(w) if substrings_in_path(s, apply_norm_to) else None, std_weights)
        # Function that applies settings.norm_fn to every leaf of the params dictionary
        # The result is a dictionary that contains the normed params
        norm_fn =  jax.jit(lambda tree : tree_map_with_path(lambda s,w,std : get_norm_fn(args.norm.norm_fn)(w,args.norm.norm_multiply,std) if substrings_in_path(s,apply_norm_to) else w,tree,target_std))
    elif args.norm.norm_fn == "global_center_std_uncenter":
        # Get the standard deviations of the weights in the beginning
        target_std = tree_map_with_path(lambda s,w : jax.vmap(lambda x : jnp.std(x,keepdims=True),in_axes=(0,))(w) if substrings_in_path(s,apply_norm_to) else None, std_weights)
        # Function that applies settings.norm_fn to every leaf of the params dictionary
        # The result is a dictionary that contains the normed params
        norm_fn =  jax.jit(lambda tree : tree_map_with_path(lambda s,w,std : get_norm_fn(args.norm.norm_fn)(w,args.norm.norm_multiply,std) if substrings_in_path(s,apply_norm_to) else w,tree,target_std))
    else:
        # Function that applies settings.norm_fn to every leaf of the params dictionary
        # The result is a dictionary that contains the normed params
        norm_fn =  jax.jit(lambda tree : tree_map_with_path(lambda s,w : get_norm_fn(args.norm.norm_fn)(w,args.norm.norm_multiply) if substrings_in_path(s,apply_norm_to) else w,tree))

    # We want to be able to specify how much the weights are changed via:
    # new_params = (1-change_scale)*params + change_scale*params_normed
    # If change_scale is not provided via settings, we simply set it to 1. Otherwise change scale is a function that takes:
    # n -> current step
    # N -> Max steps
    # l -> current layer
    # L -> Max layers
    change_scale = get_change_scale(args.norm.change_scale)

    # This function calculates the new params as described earlier
    def change_fn(w,normed_w,n,N,l,L):
        s = change_scale(n,N,l,L)
        return (1-s)*w + s*normed_w

    # This function takes as input the params, the normed params, n, N, the dictionary containing the layer depth and L
    # change_fn is then applied to every common leaf of params, normed_params and the layer depth dictionary 
    layerwise_stepscale_fn = jax.jit(lambda params,normed_params,n,N,layer_depth_dict,L : 
                                        tree_map_with_path(lambda s,w,normed_w,l : change_fn(w,normed_w,n,N,l,L) 
                                                        if substrings_in_path(s,apply_norm_to) else w,params,normed_params,layer_depth_dict))

    grads = []
    weights = layerwise_stepscale_fn(weights,norm_fn(weights),1,args.num_steps,layer_depth_dict,num_layers)
    
    NUM_STEPS = 250

    grads_mean = None
    grads_var = None
    for i,(img,lbl) in zip(range(NUM_STEPS),ds_train):

        # Generate new random keys for this step
        keys = jax.random.split(split_key,num=args.num_devices*args.num_experiments_per_device+1)
        sk,split_key = keys[:-1],keys[-1]
                    
        grad,aux = get_grad_fn(weights,batch_stats,img,lbl,sk,model.apply)

        if grads_mean == None:
            grads_mean = {key : val["kernel"] for key,val in grad.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))}
            grads_var = {key : tree_map(lambda x : x**2,val["kernel"]) for key,val in grad.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))}
        else:
            grads_mean = tree_map(lambda g1,g2 : g1+g2,grads_mean,{key : val["kernel"] for key,val in grad.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))})
            grads_var = tree_map(lambda g1,g2 : g1+g2,grads_var,{key : tree_map(lambda x : x**2,val["kernel"]) for key,val in grad.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))})
    
    grads_mean = tree_map(lambda x : x/NUM_STEPS,grads_mean)
    grads_var = tree_map(lambda x : x/NUM_STEPS,grads_var)

    mu = optimizer_state[1][0].mu
    nu = optimizer_state[1][0].nu

    mu = {key : val["kernel"] for key,val in mu.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))}
    nu = {key : val["kernel"] for key,val in nu.items() if  substrings_in_path(key.lower(),"|".join(["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]))}

    def transform(measures):
        measures = tree_map(lambda x : x.reshape(3,-1,x.shape[-1]),measures)
        return measures


    @jax.jit
    @partial(jax.vmap,in_axes=(0,0))
    def cosine_sim_fn(t1,t2):
        cosine_sim = (jnp.sum(t1*t2,axis=0))/(jnp.linalg.vector_norm(t1,axis=0)*jnp.linalg.vector_norm(t2,axis=0)+1e-8)
        return cosine_sim
    
    @jax.jit
    @partial(jax.vmap,in_axes=(0,0))
    def dist_fn(t1,t2):
        return jnp.linalg.vector_norm(t1-t2,axis=0) 

    measure = {}

    mu = transform(mu)
    nu = transform(nu)
    grads_mean = transform(grads_mean)
    grads_var = transform(grads_var)

    for key in grads_mean.keys():
        measure[key] = {"mu" : {"cos" : tree_map(lambda t1,t2 : cosine_sim_fn(t1,t2),mu[key],grads_mean[key]),
                                   "dist" :  tree_map(lambda t1,t2 : dist_fn(t1,t2),mu[key],grads_mean[key])}, 
                        "nu" : {"cos" : tree_map(lambda t1,t2 : cosine_sim_fn(t1,t2),nu[key],grads_var[key]),
                                   "dist" :  tree_map(lambda t1,t2 : dist_fn(t1,t2),nu[key],grads_var[key])}}

    weights,optimizer_state,updates = update_states_fn(weights, grad, optimizer_state, optimizer.update)

    if args.norm.reverse_norms:
        weights = reverse_norms(weights)

    # Save weights,batch_stats and optim state

    with open(os.path.join(args.save_path,"adam_drift",ckpt), "wb") as f:
        pkl.dump(measure,f)