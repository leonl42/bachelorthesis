import jax
import jax.numpy as jnp
import argparse
import pickle as pkl
from jax.sharding import Mesh, PartitionSpec as P,NamedSharding
from jax.experimental import mesh_utils
from utils import *
import json
from tqdm import tqdm
import os
from types import SimpleNamespace
from utils import train_step
import copy
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('save_path', type=str)
parser.add_argument('--reset', action=argparse.BooleanOptionalAction)
parse_args = parser.parse_args()

with open(parse_args.save_path + "settings.json", "r") as f:
    args_dict = json.load(f)
args = dict_to_namespace(args_dict)

args.save_path = parse_args.save_path

if parse_args.reset:
    args.at_step = 0
    if os.path.exists(args.save_path + "states/"):
        shutil.rmtree(args.save_path + "states/")

    if os.path.exists(args.save_path + "grads/"):
        shutil.rmtree(args.save_path + "grads/")

    if os.path.exists(args.save_path + "hessians/"):
        shutil.rmtree(args.save_path + "hessians/")

    if os.path.exists(args.save_path + "train_stats/"):
        shutil.rmtree(args.save_path + "train_stats/")

    if os.path.exists(args.save_path + "test_stats/"):
        shutil.rmtree(args.save_path + "test_stats/")

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

ds_train,ds_test = get_dataset(args)

model,num_conv_layer = get_model(args)

# Get key paths of all layers
with jax.default_device(default_cpu_device):
    helper_weights = model.init(jax.random.key(0),jnp.ones((1,32,32,3)))["params"]

optimizer = get_optimizer(args,helper_weights=helper_weights)


# Initialize model 
if args.at_step == 0:
    keys = jax.random.split(split_key,num=args.num_devices*args.num_experiments_per_device+1)
    sk,split_key = keys[:-1],keys[-1]
    weights,batch_stats,optimizer_state = get_states_device_put(model.init,optimizer.init,jnp.ones((1,32,32,3)),sk,default_cpu_device,named_sharding)
else:
    with open(args.save_path + "states/" + str(args.at_step) + ".pkl", "rb") as f:
        weights,batch_stats,optimizer_state = pkl.load(f)

norm_fn = get_norm_fn(args.norm.norm_fn)

for i,(img,lbl) in zip(tqdm(range(args.at_step,args.num_steps+1)),ds_train):

    # Generate new random keys for this step
    keys = jax.random.split(split_key,num=args.num_devices*args.num_experiments_per_device+1)
    sk,split_key = keys[:-1],keys[-1]

    if i%args.norm.norm_every == 0 and args.norm.norm_every != -1:
        weights = norm_fn(weights,args.norm.norm_multiply,i,args.num_steps,num_conv_layer)

    if i%args.optimizer.apply_wd_every == 0 and args.optimizer.apply_wd_every != -1:
        pass

    # Perform a single train step
    #weights,batch_stats,optimizer_state,aux = train_step(weights,batch_stats,optimizer_state,img,lbl,sk,model.apply,optimizer.update)


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
            json.dump(namespace_to_dict(copy.deepcopy(args)),f)

    if i%args.save_args.save_train_stats_every == 0 and args.save_args.save_train_stats_every != -1:
        with open(args.save_path + "train_stats/" + str(i) + ".pkl", "wb") as f:
            pkl.dump({"loss" : aux["loss"], "acc" : aux["acc"]},f)
            
    if i%args.save_args.save_test_stats_every == 0 and args.save_args.save_test_stats_every != -1:
        with open(args.save_path + "test_stats/" + str(i) + ".pkl", "wb") as f:
            pkl.dump(eval(weights,batch_stats,model.apply,ds_test,78),f)
