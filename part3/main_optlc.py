
  
import jax
#jax.config.update('jax_platform_name', 'cpu')
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
from itertools import chain
from jax.numpy.linalg import matrix_norm

import os
from main import SimpleNamespaceNone,FullyConnected

def get_layer_cushion(params,x0):
    x0 = x0.reshape(x0.shape[0],-1)

    x1 = nn.Dense(features=512).apply({"params" : params["params"]["Dense_0"]},x0)
    y1 = nn.relu(x1)

    x2 = nn.Dense(features=256).apply({"params" : params["params"]["Dense_1"]},y1)
    y2 = nn.relu(x2)

    x3 = nn.Dense(features=10).apply({"params" : params["params"]["Dense_2"]},y2)

    lc1 = vector_norm(x1,ord=2,axis=-1)/(matrix_norm(params["params"]["Dense_0"]["kernel"],ord="fro")*vector_norm(x0,ord=2,axis=-1) +1e-7)
    lc2 = vector_norm(x2,ord=2,axis=-1)/(matrix_norm(params["params"]["Dense_1"]["kernel"],ord="fro")*vector_norm(y1,ord=2,axis=-1) +1e-7)
    lc3 = vector_norm(x3,ord=2,axis=-1)/(matrix_norm(params["params"]["Dense_2"]["kernel"],ord="fro")*vector_norm(y2,ord=2,axis=-1) +1e-7)

    return lc1,lc2,lc3

@partial(jax.vmap, in_axes=(0,None))
@jax.jacobian
def get_jacobian(x,fn):
    return fn(x)

def get_interlayer_cushion(params,x0):
    x0 = x0.reshape(x0.shape[0],-1)

    dense1 = lambda x : nn.Dense(features=512).apply({"params" : params["params"]["Dense_0"]},x)
    dense2 = lambda x : nn.Dense(features=256).apply({"params" : params["params"]["Dense_1"]},x)
    dense3 = lambda x : nn.Dense(features=10).apply({"params" : params["params"]["Dense_2"]},x)

    M11 = lambda x : x
    M12 = lambda x : dense2(nn.relu(x))
    M13 = lambda x : dense3(nn.relu(dense2(nn.relu(x))))
    M22 = lambda x : x
    M23 = lambda x : dense3(nn.relu(x))
    M33 = lambda x : x

    # Shape (batch_size x jacobian_size)
    J11 = get_jacobian(dense1(x0),M11)
    J12 = get_jacobian(dense1(x0),M12)
    J13 = get_jacobian(dense1(x0),M13)
    J22 = get_jacobian(dense2(nn.relu(dense1(x0))),M22)
    J23 = get_jacobian(dense2(nn.relu(dense1(x0))),M23)
    J33 = get_jacobian(dense3(nn.relu(dense2(nn.relu(dense1(x0))))),M33)

    ilc11 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J11,dense1(x0))
    ilc12 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J12,dense1(x0))
    ilc13 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J13,dense1(x0))
    ilc22 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J22,dense2(nn.relu(dense1(x0))))
    ilc23 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J23,dense2(nn.relu(dense1(x0))))
    ilc33 = jax.vmap(lambda j,x : vector_norm(j@x)/(matrix_norm(j,ord="fro")*vector_norm(x) +1e-7), in_axes=(0,0))(J33,dense3(nn.relu(dense2(nn.relu(dense1(x0))))))

    return ilc11,ilc12,ilc13,ilc22,ilc23,ilc33

@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None,None))
def get_loss_fn_vmapped(params,x,y,apply_fn,loss_lc_scale):
    return get_loss_fn(params,x,y,apply_fn,loss_lc_scale)

def get_loss_fn(params,x,y,apply_fn,loss_lc_scale):
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
    loss_task = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(logits=prediction,labels=y))
    loss_lc = 0.0
    for e in chain(get_layer_cushion(params,x),get_interlayer_cushion(params,x)):
        loss_lc += -loss_lc_scale*jnp.min(e)
    loss = loss_task + loss_lc
    # Compute accuracy
    acc = jnp.mean(prediction.argmax(-1)==y)

    return loss,loss_task,loss_lc,acc


@partial(jax.grad,argnums=0,has_aux=True)
def get_grad_fn(params,x,y,apply_fn,loss_lc_scale):
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
        
    loss,loss_task,loss_lc,acc = get_loss_fn(params,x,y,apply_fn,loss_lc_scale)
    return loss, {"loss" : loss,"loss_task" : loss_task, "loss_lc" : loss_lc, "acc" : acc}

@partial(jax.jit,static_argnums=(4,5))
@partial(jax.vmap,in_axes=(0,0,0,0,None,None,None))
def step_fn(params,opt_params,x,y,apply_fn,optim_update_fn,loss_lc_scale):
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
    grad,aux = get_grad_fn(params,x,y,apply_fn,loss_lc_scale)
    
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


def eval(params,apply_fn,ds,loss_lc_scale):
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
        loss,loss_task,loss_lc,acc = get_loss_fn_vmapped(params,x_test,y_test,apply_fn,loss_lc_scale)
        stats_agg.append((loss,loss_task,loss_lc,acc))
    # Since we perform multiple steps, we have to take the average over all steps performed.
    # The resulting loss/accuracy vector will have shape (n,), where n is settings.num_parallel_exps
    loss = np.asarray(jnp.mean(jnp.stack([e[0] for e in stats_agg],axis=0),axis=0))
    loss_task = np.asarray(jnp.mean(jnp.stack([e[1] for e in stats_agg],axis=0),axis=0))
    loss_lc = np.asarray(jnp.mean(jnp.stack([e[2] for e in stats_agg],axis=0),axis=0))
    acc = np.asarray(jnp.mean(jnp.stack([e[3] for e in stats_agg],axis=0),axis=0))

    return loss,loss_task,loss_lc,acc

def train(save_path,settings):
    
    if os.path.isfile(save_path+ "stats.pkl"):
        print("Skipping: ", save_path)
        return
    
    print("Running: ", save_path)

    save_path += "/"

    os.makedirs(save_path,exist_ok=True)
    os.makedirs(save_path + "states/conv/",exist_ok=True)
    os.makedirs(save_path + "states/dense/",exist_ok=True)
    os.makedirs(save_path + "states/optim/",exist_ok=True)
    
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
    # infinite iterator for training a certain number of steps
    ds_train_train = [ds_train.repeat(-1).shuffle(25000).batch(64).prefetch(64).as_numpy_iterator() for _ in range(settings.num_parallel_exps)]

    # non infinite iterator to evaluate the model on the train set
    ds_train_eval = [ds_train.shuffle(25000).batch(1000).prefetch(64) for _ in range(settings.num_parallel_exps)]
    # non infinite iterator to evaluate the model on the test set
    ds_test_eval = [ds_test.shuffle(5000).batch(1000).prefetch(64) for _ in range(settings.num_parallel_exps)]

    stats_ckpts = {"train_acc" : {}, "test_acc" : {}, "train_loss" : {}, "test_loss" : {}, "train_loss_task" : {}, "train_loss_lc" : {}, "test_loss_task" : {}, "test_loss_lc" : {}}

    # Initialize the model and ensure that each model is initialized with a different random seed
    model = FullyConnected(10,nn.relu)
    split_key = jax.random.key(random.randint(1,2473673438))
    use_keys = jax.random.split(key=split_key,num=settings.num_parallel_exps)
    params = init_model(use_keys,jnp.ones((1,32,32,3)),model.init)

    # Set base optimizer
    if settings.optim == "sgdm":
        optim = optax.sgd(learning_rate=settings.lr)
        if settings.wd:
            optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"dense","kernel"),params)),optim)
    
    elif settings.optim == "adam":
        optim = optax.adam(learning_rate=settings.lr)
        if settings.wd:
            optim = optax.chain(optax.add_decayed_weights(weight_decay=settings.wd,mask=tree_map_with_path(lambda s,_ : substrings_in_path(s,"dense","kernel"),params)),optim)
    
    elif settings.optim == "adamw":
        if settings.wd:
            optim = optax.adamw(learning_rate=settings.lr,weight_decay=settings.wd)
        else:
            optax.adamw(learning_rate=settings.lr)

    else:
        raise Exception("Optim not known")
        

    opt_params = init_optimizer(params,optim.init)

    # Perform "settings.steps" on a dataset that is an infinite iterator
    for i,(x_train,y_train)in zip(range(settings.steps+1),ds_stack_iterator(*ds_train_train)):

        # Save dense params
        if settings.save_dense_every and random.randint(a=1,b=settings.save_dense_every) == 1:
            with open(save_path+"states/dense/"+str(i)+".pkl","wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,params),f)

        # Save optimizer params
        if settings.save_optim_every and random.randint(a=1,b=settings.save_optim_every) == 1:
            with open(save_path+"states/optim/"+str(i)+".pkl","wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,opt_params),f)

        # Perform the gradient update step
        params,opt_params,_ = step_fn(params,opt_params,x_train,y_train,model.apply,optim.update,settings.loss_lc_scale)        
        
        if settings.eval_every and random.randint(a=1,b=settings.eval_every) == 1:

            train_loss,train_loss_task,train_loss_lc,train_acc = eval(params,model.apply,ds_train_eval,settings.loss_lc_scale)
            test_loss,test_loss_task,test_loss_lc,test_acc = eval(params,model.apply,ds_test_eval,settings.loss_lc_scale)

            stats_ckpts["train_loss"][i] = train_loss
            stats_ckpts["train_loss_task"][i] = train_loss_task
            stats_ckpts["train_loss_lc"][i] = train_loss_lc
            stats_ckpts["train_acc"][i] = train_acc

            stats_ckpts["test_loss"][i] = test_loss
            stats_ckpts["test_loss_task"][i] = test_loss_task
            stats_ckpts["test_loss_lc"][i] = test_loss_lc
            stats_ckpts["test_acc"][i] = test_acc

            print("Setting: ", settings.loss_lc_scale, "| step: ",i, " [train_acc: ", train_acc, "| test_acc: ",test_acc, " | train_loss_task: ", train_loss_task, " | train_loss_lc: ", train_loss_lc, "]")
            # Save stats (train/test loss/accuracy)
            with open(save_path+"stats.pkl","wb") as f:
                pkl.dump(stats_ckpts,f)

if __name__ == "__main__":
    for loss_lc_scale in [0.25,0.2,0.5,0.8]:
        save_path = "./sidequest/optimlc/"+str(loss_lc_scale)+"/run_1/"
        train(save_path, SimpleNamespaceNone(num_parallel_exps=2,
                                                    steps=150000,
                                                    lr=0.0001,
                                                    optim="adam",
                                                    eval_every=1000,
                                                    save_dense_every=1000,
                                                    loss_lc_scale = loss_lc_scale))