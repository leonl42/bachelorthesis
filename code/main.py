import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
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
import os
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str)
parser.add_argument('--reset', action=argparse.BooleanOptionalAction)
parser.add_argument('--bfloat16', action=argparse.BooleanOptionalAction)
parser.add_argument("--overwrite-num-steps", type=int, default=None)
parser.add_argument("--overwrite-save-state", type=int, default=None)
parser.add_argument("--overwrite-save-grad", type=int, default=None)
parse_args = parser.parse_args()

with open(parse_args.save_path + "settings.json", "r") as f:
    args_dict = json.load(f)

if parse_args.overwrite_num_steps:
    args_dict["num_steps"] = parse_args.overwrite_num_steps

if parse_args.overwrite_save_state:
    args_dict["save_args"]["save_states_every"] = parse_args.overwrite_save_state

if parse_args.overwrite_save_grad:
    args_dict["save_args"]["save_grad_every"] = parse_args.overwrite_save_grad

if parse_args.overwrite_num_steps or parse_args.overwrite_save_state or parse_args.overwrite_save_grad:
    with open(parse_args.save_path + "settings.json", "w") as f:
        json.dump(args_dict,f,indent=4)

args = dict_to_namespace(args_dict)
args.save_path = parse_args.save_path

if parse_args.reset:
    args.at_step = 0
    if os.path.exists(args.save_path + "states"):
        shutil.rmtree(args.save_path + "states")

    if os.path.exists(args.save_path + "grads"):
        shutil.rmtree(args.save_path + "grads")
        
    if os.path.exists(args.save_path + "hessians"):
        shutil.rmtree(args.save_path + "hessians")

    if os.path.exists(args.save_path + "train_stats"):
        shutil.rmtree(args.save_path + "train_stats")

    if os.path.exists(args.save_path + "test_stats"):
        shutil.rmtree(args.save_path + "test_stats")

os.makedirs(args.save_path + "states/", exist_ok=True)
os.makedirs(args.save_path + "grads/", exist_ok=True)
os.makedirs(args.save_path + "hessians/", exist_ok=True)
os.makedirs(args.save_path + "train_stats/", exist_ok=True)
os.makedirs(args.save_path + "test_stats/", exist_ok=True)

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


# Initialize model 
if args.at_step == 0:
    keys = jax.random.split(split_key,num=args.num_devices*args.num_experiments_per_device+1)
    sk,split_key = keys[:-1],keys[-1]
    weights,batch_stats,optimizer_state = get_states(model.init,optimizer.init,jnp.ones((1,32,32,3)),sk,default_cpu_device)

    if args.save_args.save_states_every != -1:
        with open(args.save_path + "states/" + str(0) + ".pkl", "wb") as f:
            pkl.dump((weights,batch_stats,optimizer_state),f)

else:
    with open(args.save_path + "states/" + str(args.at_step) + ".pkl", "rb") as f:
        weights,batch_stats,optimizer_state = pkl.load(f)

weights,batch_stats,optimizer_state = device_put(named_sharding,weights,batch_stats,optimizer_state)

if parse_args.bfloat16:
    weights = tree_map(lambda x : jnp.astype(x,jnp.bfloat16),weights)
    optimizer_state = tree_map(lambda x : jnp.astype(x,jnp.bfloat16) if jnp.isdtype(x,jnp.float32) else x,optimizer_state)

if args.norm.norm_fn == "center_std_uncenter" or args.norm.norm_fn == "global_center_std_uncenter":
    if args.at_step == 0:
        std_weights = weights
    else:
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

print("Running: {0}".format(parse_args.save_path))

for i,(img,lbl) in zip(tqdm(range(args.at_step+1,args.num_steps+1)),ds_train):

    # Generate new random keys for this step
    keys = jax.random.split(split_key,num=args.num_devices*args.num_experiments_per_device+1)
    sk,split_key = keys[:-1],keys[-1]

    if args.norm.start_after is None or i>=args.norm.start_after:
        if args.norm.stop_after is None or i<=args.norm.stop_after:
            if i%args.norm.norm_every == 0 and args.norm.norm_every != -1:
                weights = layerwise_stepscale_fn(weights,norm_fn(weights),i,args.num_steps,layer_depth_dict,num_layers)

    if i%args.optimizer.apply_wd_every == 0 and args.optimizer.apply_wd_every != -1:
        pass

    grad,aux = get_grad_fn(weights,batch_stats,img,lbl,sk,model.apply)
    batch_stats = aux["batch_stats"]
    weights,optimizer_state = update_states_fn(weights, grad, optimizer_state, optimizer.update)

    if args.optimizer.apply_wd_every != -1:
        pass

    if args.norm.reverse_norms:
        weights = reverse_norms(weights)

    # Save weights,batch_stats and optim state
    if i%args.save_args.save_grad_every == 0 and args.save_args.save_grad_every != -1:
        with open(args.save_path + "grads/" + str(i) + ".pkl", "wb") as f:
            pkl.dump(grad,f)
    if i%args.save_args.save_hessian_every == 0 and args.save_args.save_grad_every != -1:
        pass
    if i%args.save_args.save_states_every == 0 and args.save_args.save_states_every != -1:
        with open(args.save_path + "states/" + str(i) + ".pkl", "wb") as f:
            pkl.dump((weights,batch_stats,optimizer_state),f)

        args.at_step = i
        with open(args.save_path + "settings.json", "w") as f:
            json.dump(namespace_to_dict(args),f,indent=4)
    
    if i%args.save_args.save_train_stats_every == 0 and args.save_args.save_train_stats_every != -1:
        with open(args.save_path + "train_stats/" + str(i) + ".pkl", "wb") as f:
            pkl.dump({"loss" : aux["loss"], "acc" : aux["acc"]},f)
            
    if i%args.save_args.save_test_stats_every == 0 and args.save_args.save_test_stats_every != -1:
        with open(args.save_path + "test_stats/" + str(i) + ".pkl", "wb") as f:
            pkl.dump(eval(weights,batch_stats,model.apply,ds_test,78),f)
