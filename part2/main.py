
  
import os
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = "0.5"
import jax
jax.numpy.ones((32,4))*jax.numpy.zeros((32,4))
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
  tf.config.experimental.set_memory_growth(gpu, True)

import random
import jax.numpy as jnp
import flax.linen as nn
from functools import partial
import optax
from jax.tree_util import keystr,tree_map_with_path,tree_map
import numpy as np
import tensorflow_datasets as tfds
from tqdm import tqdm
import pickle as pkl
from types import SimpleNamespace

class SimpleNamespaceNone(SimpleNamespace):
    # Returns None instead of throwing an error when an undefined name is accessed
    def __getattr__(self, _):
        return None

class VGG11(nn.Module):
    num_classes: int
    activation_fn: any

    @nn.compact
    def __call__(self, x ,train: bool = True):
        x = nn.Conv(features=64,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=128,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=256,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=256,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=True)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.avg_pool(x,window_shape=(1,1),strides=(1,1))

        x = x.reshape(x.shape[0],-1)

        x = nn.Dense(features=self.num_classes,name="out")(x)

        return x


@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_center_normalize(w,scale):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
    Returns: 
        w <jax.Array> : Weight matrix but with the channels means normalized to 0 and the channel norms normalized to "scale".
    """
    shape = w.shape
    n_dims = len(shape)

    # Compute the channel means
    mean = jnp.mean(w,axis=tuple(range(n_dims-1)),keepdims=True)

    # Compute the weight matrix with channel means normalized to 0
    w = (w-mean)

    # Compute the channel norms
    norm = jnp.expand_dims(jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=False),axis=tuple(range(n_dims-1)))

    # Compute the weight matrix with channel norms normalized to "scale"
    w = scale*w/(norm+1e-7)

    return w

@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_center_normalize_uncenter(w,scale):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
    Returns: 
        w <jax.Array> : Weight matrix but with the channels means normalized to 0, the channel norms normalized to "scale" and channel means scaled back to original mean.
    """
    shape = w.shape
    n_dims = len(shape)

    # Compute the channel means
    mean = jnp.mean(w,axis=tuple(range(n_dims-1)),keepdims=True)

    # Compute the weight matrix with channel means normalized to 0
    w = (w-mean)

    # Compute the channel norms
    norm = jnp.expand_dims(jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=False),axis=tuple(range(n_dims-1)))

    # Compute the weight matrix with channel norms normalized to "scale"
    w = scale*w/(norm+1e-7) + mean

    return w

@jax.jit
@partial(jax.vmap,in_axes=(0,None,0))
def weight_center_std_uncenter(w,scale,target_std):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
        std <float> : target standard deviation
    Returns: 
        w <jax.Array> : Weight matrix but with the channels means normalized to 0 and the channel norms normalized to "scale*target_std".
    """
    shape = w.shape
    n_dims = len(shape)

    # Compute the channel means
    mean = jnp.mean(w,axis=tuple(range(n_dims-1)),keepdims=True)

    # Compute the weight matrix with channel means normalized to 0
    w = (w-mean)

    # Compute the channel stds
    std = jnp.std(w,axis=tuple(range(n_dims-1)),keepdims=True)

    # Compute the weight matrix with channel norms normalized to "scale"
    w = scale*target_std*w/(std+1e-7) + mean

    return w

@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_reverse_center_normalize(w,scale):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
    Returns: 
        w <jax.Array> : Weight matrix but with the input means normalized to 0 and the channel norms normalized to "scale".
    """
    shape = w.shape
    n_dims = len(shape)

    # Compute the channel means
    if n_dims == 2:
        mean = jnp.mean(w,axis=1,keepdims=True)
    else:
        mean = jnp.mean(w,axis=(0,1,3),keepdims=True)
    # Compute the weight matrix with channel means normalized to 0
    w = (w-mean)

    # Compute the channel norms
    norm = jnp.expand_dims(jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=False),axis=tuple(range(n_dims-1)))

    # Compute the weight matrix with channel norms normalized to "scale"
    w = scale*w/(norm+1e-7)

    return w

@jax.jit
@partial(jax.vmap,in_axes=(0,None))
def weight_normalize(w,scale):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
    Returns: 
        w <jax.Array> : Weight matrix but with the channel norms normalized to "scale".
    """

    shape = w.shape
    n_dims = len(shape)

    # Compute the channel norms
    norm = jnp.expand_dims(jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=False),axis=tuple(range(n_dims-1)))

    # Compute the weight matrix with channel norms normalized to "scale"
    w = scale*w/(norm+1e-7)

    return w


@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None))
def get_loss_fn_vmapped(params,x,y,apply_fn):
    return get_loss_fn(params,x,y,apply_fn)

def get_loss_fn(params,x,y,apply_fn):
    """
    Performes a forward pass (input -> prediction)

    Takes:
        params <dict> : Weights of the model
        x <jax.Array> : Input to the model
        y <jax.Array> : target 
        apply_fn <function> : function that predicts an output given an input (model.apply)
    
    Returns:
        loss <jax.Array> : loss of the model
        accuracy <jax.Array> : accuracy of the model
    """

    prediction = apply_fn(params,x)

    # Compute loss
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits=prediction,labels=y))
    # Compute accuracy
    acc = jnp.mean(prediction.argmax(-1)==y)

    return loss,acc


@partial(jax.grad,argnums=0,has_aux=True)
def get_grad_fn(params,x,y,apply_fn):
    """
    Calcualtes the grad of the loss w.r.t. to the parameters

    Takes:
        params <dict> : Weights of the model
        x <jax.Array> : Input to the model
        y <jax.Array> : target 
        apply_fn <function> : function that predicts an output given an input (model.apply)
    
    Returns:
        grad <jax.Array> : Gradient of the loss w.r.t. to the paremeters
        aux <dict> : auxiliary data e.g. loss and accuracy
    """
        
    loss,acc = get_loss_fn(params,x,y,apply_fn)
    return loss, {"loss" : loss,"acc" : acc}

@partial(jax.jit,static_argnums=(4,5))
@partial(jax.vmap,in_axes=(0,0,0,0,None,None))
def step_fn(params,opt_params,x,y,apply_fn,optim_update_fn):
    """
    Updates the model and optimizer weights

    Takes:
        params <dict> : Weights of the model
        opt_params <dict> : Weights of the optimizer
        x <jax.Array> : Input to the model
        y <jax.Array> : target 
        apply_fn <function> : function that predicts an output given an input (model.apply)
        optim_update_fn <function> : function that produces new optimizer weights given the grad (optim.update)
    Returns:
        new_params <dict> : New weights of the model
        new_opt_params <dict> : New weights of the model
        aux <dict> : auxiliary data e.g. loss and accuracy
    """
    grad,aux = get_grad_fn(params,x,y,apply_fn)

    updates,new_opt_params = optim_update_fn(grad,opt_params,params)
    new_params = optax.apply_updates(params,updates)

    return new_params,new_opt_params,aux

def substrings_in_path(s,*substrings):
    """
    Returns True if all strings in *substrings is in s, else False. 

    Takes:
        s <string> : string for which we want to check if it has substrings inside
        *substrings <string> : list of strings for which we want to check if all of the appear in s
    Returns:
        <boolean>
    """
    return all([sub.lower() in keystr(s).lower() for sub in substrings])


@partial(jax.vmap, in_axes=(0,None,None))
def init_model(key,model_input,init_fn):
    """
    Initialize n models at the same time where n is the size of key
    """
    params = init_fn(key,model_input)
    return params

@partial(jax.vmap, in_axes=(0,None))
def init_optimizer(weights,init_fn):
    """
    Initialize n optimizers at the same time where n is the size of key
    """
    return init_fn(weights)

def ds_stack_iterator(*ds):
    """
    Takes:
        *ds <list> : list of iterators (e.g. tensorflow dataset)
    Returns:
        A generator that combines the elements of the iterators into a single Array
        For example:
            [(X1[32,32,3],Y1[10]), (X2[32,32,3],Y2[10]),..., (Xn[32,32,3],Yn[10])] -> (X[n,32,32,3],Y[n,10])

    """
    for ds_elems in zip(*ds):
        yield jnp.stack([jnp.asarray(e[0]) for e in ds_elems]),jnp.stack([jnp.asarray(e[1]) for e in ds_elems]) 


def eval(params,apply_fn,ds):
    """
    Calculates the loss and accuracy of the model over a complete dataset

    Takes:
        params <dict> : Weights of the model
        apply_fn <function> : Function that predicts an output given an input (model.apply)
        ds <generator> : Dataset
    Returns:
        loss <jax.Array> : loss
        acc <jax.Array> : accuracy 
    """
    stats_agg = []
    for x_test,y_test in ds_stack_iterator(*ds):
        loss,acc = get_loss_fn_vmapped(params,x_test,y_test,apply_fn)
        stats_agg.append((loss,acc))
    # Since we perform multiple steps, we have to take the average over all steps performed.
    # The resulting loss/accuracy vector will have shape (n,), where n is settings.num_parallel_exps
    loss = np.asarray(jnp.mean(jnp.stack([e[0] for e in stats_agg],axis=0),axis=0))
    acc = np.asarray(jnp.mean(jnp.stack([e[1] for e in stats_agg],axis=0),axis=0))

    return loss,acc

def train(save_path,settings):
    
    if os.path.isfile(os.path.join(save_path,"stats.pkl")):
        print("Skipping: ", save_path)
        return
    
    print("Running: ", save_path)

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(os.path.join(save_path,"states","models"),exist_ok=True)
    os.makedirs(os.path.join(save_path,"states","optim"),exist_ok=True)
    #####################################
        ## Initialize the dataset ##
    #####################################
    builder = tfds.builder("cifar10",data_dir="./datasets")
    builder.download_and_prepare()
    ds_train,ds_test = builder.as_dataset(split=["train", "test"])

    # Transform dataset from dict into tuple
    solve_dict = lambda elem : (elem["image"],elem["label"])
    ds_train,ds_test = ds_train.map(solve_dict),ds_test.map(solve_dict)

    # Cast image dtype to bfloat16
    dtype = tf.dtypes.bfloat16 if settings.bfloat16 else tf.dtypes.float32
    cast = lambda img,lbl : (tf.cast(img,dtype),lbl)
    ds_train,ds_test = ds_train.map(cast),ds_test.map(cast)

    # Normalize images with precalculated mean and std
    mean = tf.convert_to_tensor([0.32768, 0.32768, 0.32768],dtype=dtype)[None,None,:]
    std = tf.convert_to_tensor([0.27755222, 0.26925606, 0.2683012 ],dtype=dtype)[None,None,:]
    normalize = lambda img,lbl : ((img/255-mean)/std,lbl)

    # Prepare a number of "settings.num_parallel_exps" independent datasets for training
    ds_train,ds_test = ds_train.map(normalize),ds_test.map(normalize)
    # infinite iterator for training a certain number of steps
    ds_train_train = [ds_train.repeat(-1).shuffle(25000).batch(64).prefetch(64).as_numpy_iterator() for _ in range(settings.num_parallel_exps)]

    # non infinite iterator to evaluate the model on the train set
    ds_train_eval = [ds_train.shuffle(25000).batch(1000).prefetch(64) for _ in range(settings.num_parallel_exps)]
    # non infinite iterator to evaluate the model on the test set
    ds_test_eval = [ds_test.shuffle(5000).batch(1000).prefetch(64) for _ in range(settings.num_parallel_exps)]

    stats_ckpts = {"train_acc" : {}, "test_acc" : {}, "train_loss" : {}, "test_loss" : {}}

    # Initialize the model and ensure that each model is initialized with a different random seed
    model = VGG11(10,nn.relu)
    split_key = jax.random.key(random.randint(1,2473673438))
    # Set seed for pythons random for determining time steps where the model is saved and evaluated
    random.seed(42)
    use_keys = jax.random.split(key=split_key,num=settings.num_parallel_exps)
    params = init_model(use_keys,jnp.ones((1,32,32,3)),model.init)

    # Set base optimizer
    if settings.optim == "sgdm":
        optim = optax.sgd(learning_rate=settings.lr)
        if settings.wd:
            optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"kernel"),params)),optim)
    
    elif settings.optim == "adam":
        optim = optax.adam(learning_rate=settings.lr)
        if settings.wd:
            optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"kernel"),params)),optim)
    
    elif settings.optim == "adamw":
        if settings.wd:
            optim = optax.adamw(learning_rate=settings.lr,weight_decay=settings.wd)
        else:
            optax.adamw(learning_rate=settings.lr)

    else:
        raise Exception("Optim not known")

    opt_params = init_optimizer(params,optim.init)

    if settings.bfloat16:
        params = tree_map(lambda x : jnp.astype(x,jnp.bfloat16),params)
        opt_params = tree_map(lambda x : jnp.astype(x,jnp.bfloat16) if jnp.isdtype(x,jnp.float32) else x,opt_params)

    # If we want to perform normalization/rescaling, initialize the transform
    if settings.norm_every:
        # If we want to use the normalization scheme proposed by Niehaus et al. 2024, we have to calculate the standard deviation before training
        if settings.norm_fn == weight_center_std_uncenter:
            # Get the standard deviations of the weights in the beginning
            target_std = tree_map_with_path(lambda s,w : jax.vmap(lambda x : jnp.std(x,axis=tuple(range(len(x.shape)-1)),keepdims=True),in_axes=(0,))(w) if substrings_in_path(s,"kernel") else None, params)
            # Function that applies settings.norm_fn to every leaf of the params dictionary
            # The result is a dictionary that contains the normed params
            norm_fn =  jax.jit(lambda tree,n,N : tree_map_with_path(lambda s,w,std : settings.norm_fn(w,settings.norm_scale(n,N),std) if substrings_in_path(s,"kernel") else w,tree,target_std))
        else:
            # Function that applies settings.norm_fn to every leaf of the params dictionary
            # The result is a dictionary that contains the normed params
            norm_fn =  jax.jit(lambda tree,n,N : tree_map_with_path(lambda s,w : settings.norm_fn(w,settings.norm_scale(n,N)) if substrings_in_path(s,"kernel") else w,tree))
        
        # We want to be able to specify how much the weights are changed via:
        # new_params = (1-change_scale)*params + change_scale*params_normed
        # If change_scale is not provided via settings, we simply set it to 1. Otherwise change scale is a function that takes:
        # n -> current step
        # N -> Max steps
        # l -> current layer
        # L -> Max layers
        if settings.change_scale == None:
            settings.change_scale = lambda n,N,l,L : 1

        # Assign a depth to each layer and store it in a dictionary.
        # This is useful since we can now map over the params dict and this dict at the same time and directly have the layer depth for each lef
        get_layer_depth_dict = {"params" : {"Conv_0" : {"kernel" : 1, "bias" : 0},"Conv_1" : {"kernel" : 2, "bias" : 0},"Conv_2" : {"kernel" : 3, "bias" : 0}
                                                ,"Conv_3" : {"kernel" : 4, "bias" : 0},"Conv_4" : {"kernel" : 5, "bias" : 0},"Conv_5" : {"kernel" : 6, "bias" : 0}
                                                ,"Conv_6" : {"kernel" : 7, "bias" : 0},"Conv_7" : {"kernel" : 8, "bias" : 0}, "out" : {"kernel" : 9, "bias" : 0}}}
        
        # This function calculates the new params as described earlier
        def change_fn(w,normed_w,n,N,l,L):
            change_scale = settings.change_scale(n,N,l,L)
            return (1-change_scale)*w + change_scale*normed_w
        
        # This function takes as input the params, the normed params, n, N, the dictionary containing the layer depth and L
        # change_fn is then applied to every common leaf of params, normed_params and the layer depth dictionary 
        layerwise_stepscale_fn = jax.jit(lambda params,normed_params,n,N,layer_depth_dict,L : 
                                         tree_map_with_path(lambda s,w,normed_w,l : change_fn(w,normed_w,n,N,l,L) 
                                                            if substrings_in_path(s,"kernel") else w,params,normed_params,layer_depth_dict))

    # Perform "settings.steps" on a dataset that is an infinite iterator
    for i,(x_train,y_train)in zip(tqdm(range(settings.steps+1)),ds_stack_iterator(*ds_train_train)):

        # Save model params
        if settings.save_model_every and random.randint(a=1,b=settings.save_model_every) == 1:
            with open(os.path.join(save_path,"states","model",str(i)+".pkl"), "wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,params),f)

        # Save optimizer params
        if settings.save_optim_every and random.randint(a=1,b=settings.save_optim_every) == 1:
            with open(os.path.join(save_path,"states","optim",str(i)+".pkl"), "wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,opt_params),f)

        # Perform the gradient update step
        params,opt_params,_ = step_fn(params,opt_params,x_train,y_train,model.apply,optim.update)        
        
        if settings.eval_every and random.randint(a=1,b=settings.eval_every) == 1:

            train_loss,train_acc = eval(params,model.apply,ds_train_eval)
            test_loss,test_acc = eval(params,model.apply,ds_test_eval)

            stats_ckpts["train_loss"][i] = train_loss
            stats_ckpts["train_acc"][i] = train_acc

            stats_ckpts["test_loss"][i] = test_loss
            stats_ckpts["test_acc"][i] = test_acc

            # Save stats (train/test loss/accuracy)
            with open(os.path.join(save_path,"stats.pkl"),"wb") as f:
                pkl.dump(stats_ckpts,f)
    

        if settings.norm_every and i%settings.norm_every == 0:
            params = layerwise_stepscale_fn(params,norm_fn(params,i,settings.steps),i,settings.steps,get_layer_depth_dict,9)

    

