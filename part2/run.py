
  
import jax

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)


import random
import jax.numpy as jnp
from jax.numpy.linalg import vector_norm
import flax.linen as nn
from functools import partial
import optax
import matplotlib.pyplot as plt
from jax.tree_util import keystr,tree_map_with_path,tree_map
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle as pkl
from types import SimpleNamespace

import os

class SimpleNamespaceNone(SimpleNamespace):
    # Returns None instead of throwing an error when an undefined name is accessed
    def __getattr__(self, _):
        return None

class FullyConnected(nn.Module):
    num_outputs : int
    activation_fn: any

    @nn.compact
    def __call__(self, x ,train: bool = True):
        x = nn.Conv(features=16,kernel_size=(3,3),padding=1,use_bias=False)(x)
        #x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=32,kernel_size=(3,3),padding=1,use_bias=False)(x)
        #x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=64,kernel_size=(3,3),padding=1,use_bias=False)(x)
        #x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=128,kernel_size=(3,3),padding=1,use_bias=False)(x)
        #x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=256,kernel_size=(3,3),padding=1,use_bias=False)(x)
        #x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.avg_pool(x,window_shape=(1,1),strides=(1,1))

        x = x.reshape(x.shape[0],-1)
        x = nn.Dense(features=1024)(x)
        x = self.activation_fn(x)
        x = nn.Dense(features=self.num_outputs)(x)

        return x
    
@jax.jit
@partial(jax.vmap,in_axes=(0,None,None))
def svd_scale(w,delta_shift,delta_scale):
    u,s,vt = jnp.linalg.svd(w,full_matrices=False)

    singular_value_scale = (jnp.sqrt(jnp.arange(start=1,stop=s.shape[0]+1)))*delta_scale + 1 + delta_shift
    #singular_value_scale = (jnp.arange(start=1,stop=s.shape[0]+1))*delta_scale + 1 + delta_shift
    s = s/singular_value_scale
    
    return u @ jnp.diag(s) @ vt

@jax.jit
@partial(jax.vmap,in_axes=(0,None,None))
def svd_exp_fit(w,delta_shift,delta_scale):
    u,s,vt = jnp.linalg.svd(w,full_matrices=False)

    max_sv = jnp.max(s)
    s = delta_scale*jnp.exp(s-max_sv)+delta_shift
    
    return u @ jnp.diag(s) @ vt

@jax.jit
@partial(jax.vmap,in_axes=(0,None,None))
def svd_static_exp_fit(w,delta_shift,delta_scale):
    u,s,vt = jnp.linalg.svd(w,full_matrices=False)

    s = jnp.arange(start=1,stop=s.shape[0]+1)
    s = ((0.5*s)/(s.shape[0]+1))+delta_scale*jnp.exp(s-(s.shape[0]+1))+delta_shift
    
    return u @ jnp.diag(s) @ vt

@jax.jit
def get_cap(*layers):
    norms_2 = jnp.stack([jnp.linalg.norm(l,ord=2) for l in layers])
    norms_fro = jnp.stack([jnp.linalg.norm(l,ord="fro") for l in layers])

    norms_2_prod = jnp.prod(norms_2)
    norms_fro_sum = jnp.sum(norms_fro)

    return norms_2_prod,norms_fro_sum

@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_center_normalize(w,scale):
    mean = jnp.mean(w,axis=0,keepdims=True)
    w = (w-mean)
    norm = jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=True)
    return scale*w/(norm+1e-7)

@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_normalize(w,scale):
    norm = jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=True)
    return scale*w/(norm+1e-7)


@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None))
def get_loss_fn_vmapped(params,x,y,apply_fn):
    return get_loss_fn(params,x,y,apply_fn)


def get_loss_fn(params,x,y,apply_fn):
    prediction = apply_fn(params,x)
    loss = optax.losses.softmax_cross_entropy_with_integer_labels(logits=prediction,labels=y)
    acc = prediction.argmax(-1)==y
    return jnp.mean(loss),jnp.mean(acc)


@partial(jax.grad,argnums=0,has_aux=True)
def get_grad_fn(params,x,y,apply_fn):
    loss,acc = get_loss_fn(params,x,y,apply_fn)
    return loss, {"loss" : loss,"acc" : acc}

@partial(jax.jit,static_argnums=(4,5))
@partial(jax.vmap,in_axes=(0,0,0,0,None,None))
def step_fn(params,opt_params,x,y,apply_fn,optim_update_fn):

        grad,aux = get_grad_fn(params,x,y,apply_fn)

        updates,new_opt_params = optim_update_fn(grad,opt_params,params)
        new_params = optax.apply_updates(params,updates)

        return new_params,new_opt_params,aux

def substrings_in_path(s,*substrings):
    return all([sub.lower() in keystr(s).lower() for sub in substrings])


@partial(jax.vmap, in_axes=(0,None,None))
def init_model(key,model_input,init_fn):
    params = init_fn(key,model_input)
    return params

@partial(jax.vmap, in_axes=(0,None))
def init_optimizer(weights,init_fn):
    return init_fn(weights)

def ds_stack_iterator(*ds):
    for ds_elems in zip(*ds):
        yield jnp.stack([e[0] for e in ds_elems]),jnp.stack([e[1] for e in ds_elems]) 


def train(save_path,settings):
    
    if os.path.isfile(save_path+ "stats.pkl"):
        print("Skipping: ", save_path)
        return
    
    print("Running: ", save_path)

    save_path += "/"

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path + "weights/",exist_ok=True)
    
    #####################################
        ## Initialize the dataset ##
    #####################################
    builder = tfds.builder("cifar10",data_dir="./datasets")
    builder.download_and_prepare()
    ds_train,ds_test = builder.as_dataset(split=["train", "test"])

    # Transform dataset from dict into tuple
    solve_dict = lambda elem : (elem["image"],elem["label"])
    ds_train,ds_test = ds_train.map(solve_dict),ds_test.map(solve_dict)

    # Cast image dtype to float32
    cast = lambda img,lbl : (tf.cast(img,tf.dtypes.float32),lbl)
    ds_train,ds_test = ds_train.map(cast),ds_test.map(cast)

    # Normalize images with precalculated mean and std
    mean = tf.convert_to_tensor([0.32768, 0.32768, 0.32768])[None,None,:]
    std = tf.convert_to_tensor([0.27755222, 0.26925606, 0.2683012 ])[None,None,:]
    normalize = lambda img,lbl : ((img/255-mean)/std,lbl)

    # Prepare a number of "settings.num_parallel_exps" independent datasets for training
    ds_train,ds_test = ds_train.map(normalize),ds_test.map(normalize)
    ds_train = [ds_train.repeat(-1).shuffle(25000).batch(64).prefetch(256).as_numpy_iterator() for _ in range(settings.num_parallel_exps)]
    ds_test = [ds_test.shuffle(5000).repeat(-1).batch(500).prefetch(256).as_numpy_iterator() for _ in range(settings.num_parallel_exps)]

    stats_ckpts = {"train_acc" : {}, "test_acc" : {}, "train_loss" : {}, "test_loss" : {}}

    # Initialize the model and ensure that each model is initialized with a different random seed
    model = FullyConnected(10,nn.relu)
    split_key = jax.random.key(random.randint(1,2473673438))
    use_keys = jax.random.split(key=split_key,num=settings.num_parallel_exps)
    params = init_model(use_keys,jnp.ones((1,32,32,3)),model.init)

    # Set base optimizer (Adam)
    optim = optax.adam(learning_rate=settings.lr)
    if settings.wd_dense:
        # Add weight decay for dense layers
        optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd_dense,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"dense","kernel"),params)),optim)
    if settings.wd_conv:
        # Add weight decay for conv layers
        optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd_conv,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"conv","kernel"),params)),optim)
    opt_params = init_optimizer(params,optim.init)

    # If we want to perform svd_scale, initialize the transform
    if settings.svd_scale_every: 
        # svd_scale_fn is a function that 
        # takes:
        #   - A dict containing the model parameters
        #   - A shift variable applied to the svd_scale function
        #   - A scale variable applied to the svd_scale function
        # returns:
        #   - A dict containing the model parameters, where "settings.svd_fn(w,shift,scale)" was applied to a weight in the dict, if 
        # "dense" and "kernel" appeared in the keys leading to a specific parameter
        svd_scale_fn =  jax.jit(lambda tree,shift,scale : tree_map_with_path(lambda s,w : settings.svd_fn(w,shift,scale) if substrings_in_path(s,"dense","kernel") else w,tree))
    
    # If we want to perform normalization/rescaling, initialize the transform
    if settings.norm_every:
        # Same principle as for svd_norm
        norm_fn =  jax.jit(lambda tree,scale : tree_map_with_path(lambda s,w : settings.norm_fn(w,scale) if substrings_in_path(s,"dense","kernel") else w,tree))

    # Perform "settings.steps" on a dataset that is an infinite iterator
    for i,(x_train,y_train)in zip(tqdm(range(settings.steps+1)),ds_stack_iterator(*ds_train)):


        if settings.save_params_every and i%settings.save_params_every == 0:
            # Save params and optimizer params
            with open(save_path+"states/"+str(i)+".pkl","wb") as f:
                pkl.dump({"params" : tree_map(lambda x : np.asarray(x),params), "opt_params" : tree_map(lambda x : np.asarray(x),opt_params)},f)


        # Perform the gradient update step
        params,opt_params,aux = step_fn(params,opt_params,x_train,y_train,model.apply,optim.update)

        # Save training stats (loss and acc). Since we train n runs at the same time, aux["loss"] has shape (n,).
        stats_ckpts["train_loss"][i] = np.asarray(aux["loss"])
        stats_ckpts["train_acc"][i] = np.asarray(aux["acc"])
        
        
        if settings.eval_every and i%settings.eval_every == 0:
            # Test the model on the test dataset for a certain number of steps
            stats_agg = []
            for _,(x_test,y_test) in zip(range(10),ds_stack_iterator(*ds_test)):
                loss,acc = get_loss_fn_vmapped(params,x_test,y_test,model.apply)
                stats_agg.append((loss,acc))
            # Since we perform multiple steps, we have to take the average over all steps performed.
            # The resulting loss/accuracy vector will have shape (n,), where n is settings.num_parallel_exps
            stats_ckpts["test_loss"][i] = np.asarray(jnp.mean(jnp.stack([e[0] for e in stats_agg],axis=0),axis=0))
            stats_ckpts["test_acc"][i] = np.asarray(jnp.mean(jnp.stack([e[1] for e in stats_agg],axis=0),axis=0))

        if settings.norm_every and i%settings.norm_every == 0:
            params = norm_fn(params,settings.norm_scale(i,settings.steps))

        if settings.svd_scale_every and i%settings.svd_scale_every == 0:
            if settings.norm_scale_equivalent:
                # In case "settings.norm_scale_equivalent" is set, we dont perform svd_scale, but instead we set the Dense layers parameter norms
                # to the norms that applying "svd_scale" would have resulted in. 
                # The params that would have resulted from applying "svd_scale" are stored in the local variable "dparams".
                # "dparams" is only used for extracting the norms of the weight vectors/matrices.
                dparams = svd_scale_fn(params,settings.svd_delta_shift(i,settings.steps),settings.svd_delta_scale(i,settings.steps))
                params = tree_map_with_path(lambda s,w,dw : vector_norm(dw,axis=0,keepdims=True,ord=2)*w/(vector_norm(w,axis=0,keepdims=True,ord=2)+1e-7) if substrings_in_path(s,"dense","kernel") else w ,params,dparams)    
            else:
                # In case "settings.norm_scale_equivalent" is not set, "svd_scale" is directly applied to the parameters
                params = svd_scale_fn(params,settings.svd_delta_shift(i,settings.steps),settings.svd_delta_scale(i,settings.steps))

    # Save stats (train/test loss/accuracy)
    with open(save_path+"stats.pkl","wb") as f:
        pkl.dump(stats_ckpts,f)
    

def exp_decayed(c):
    return lambda n,N: c*jnp.e**(-n/N)

def exp_increase(c):
    return lambda n,N: c*jnp.e**(n/N)


# Train model without any regularization
train("./exps/standard/standard/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250))
train("./exps/standard/standard/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250))

# Train model with Weight Decay
for wd in [0.00025 + x*0.00025 for x in range(25)]:
    train("./exps/wd/" + str(wd) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,wd_dense=wd,lr=0.001,eval_every=250))
    
    #train("./exps/wd/" + str(wd) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,wd_dense=wd,lr=0.001,eval_every=250))

# Train model with Singular Value Regularization
for shift in [0.5,1]:
    for scale in [0.005,0.01,0.015]:
        for scale_every in [50,100,150]:
            train("./exps/svd_scale/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_fn=svd_scale,svd_scale_every=scale_every))
            #train("./exps/svd_scale/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_scale_every=scale_every))


train("./exps/svd_scale_norm_equal/" + str(1) + "_" + str(0.01) + "_" + str(50) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.01,svd_fn=svd_scale,svd_scale_every=50,norm_scale_equivalent=True))

train("./exps/svd_scale/" + str(1) + "_" + str(0.0008) + "_" + str(10) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.0008,svd_fn=svd_scale,svd_scale_every=10))
train("./exps/svd_scale/" + str(1) + "_" + str(0.001) + "_" + str(10) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.001,svd_fn=svd_scale,svd_scale_every=10))
train("./exps/svd_scale/" + str(1) + "_" + str(0.002) + "_" + str(10) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.002,svd_fn=svd_scale,svd_scale_every=10))
train("./exps/svd_scale/" + str(1) + "_" + str(0.004) + "_" + str(10) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.004,svd_fn=svd_scale,svd_scale_every=10))
train("./exps/svd_scale/" + str(1) + "_" + str(0.006) + "_" + str(10) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : 1 ,svd_delta_scale = lambda n,N : 0.006,svd_fn=svd_scale,svd_scale_every=10))



for shift in [0,0.5,1]:
    for scale in [1,1.5,2]:
        for scale_every in [100]:
            train("./exps/svd_exp_fit/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_fn = svd_exp_fit,svd_scale_every=scale_every))
            #train("./exps/svd_scale/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_scale_every=scale_every))

for shift in [0,0.5,1]:
    for scale in [1,1.5,2]:
        for scale_every in [100]:
            train("./exps/svd_static_exp_fit/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_fn = svd_static_exp_fit,svd_scale_every=scale_every))
            #train("./exps/svd_scale/" + str(shift) + "_" + str(scale) + "_" + str(scale_every) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,eval_every=250,svd_delta_shift = lambda n,N : shift ,svd_delta_scale = lambda n,N : scale,svd_scale_every=scale_every))


# Train Weight Normalization (w = c/||w||)
for norm_scale in [0.1 + x*0.1 for x in range(10)]:
    for norm_every in [1,10,25,100,250]:
        train("./exps/norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=norm_every,norm_scale=lambda n,N : norm_scale,norm_fn=weight_normalize,eval_every=250))
        
        #train("./exps/norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=norm_every,norm_scale=lambda n,N : norm_scale,norm_fn=weight_normalize,eval_every=250))

# Train Weight Normalization with 0 centering (w = (w-mean(w))/||w-mean(w)||)
for norm_scale in [0.1 + x*0.1 for x in range(10)]:
    for norm_every in [1,10,25,100,250]:
        train("./exps/mean_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_1/", SimpleNamespaceNone(num_parallel_exps=5,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=norm_every,norm_scale=lambda n,N : norm_scale,norm_fn=weight_normalize,eval_every=250))
        
        #train("./exps/mean_norm/" + str(norm_scale) + "_" + str(norm_every) + "/run_2/", SimpleNamespaceNone(num_parallel_exps=3,steps=200000,wd_conv=0.0005,lr=0.001,norm_every=norm_every,norm_scale=lambda n,N : norm_scale,norm_fn=weight_normalize,eval_every=250))
