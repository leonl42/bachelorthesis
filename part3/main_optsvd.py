import os
import jax
jax.config.update('jax_platform_name', 'cpu')
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
import argparse
import os

class SimpleNamespaceNone(SimpleNamespace):
    # Returns None instead of throwing an error when an undefined name is accessed
    def __getattr__(self, _):
        return None

default_kernel_init = nn.initializers.lecun_normal()

from typing import (
  Any,
)
from collections.abc import Iterable, Sequence

import jax
from flax.linen import initializers
from flax.linen.dtypes import promote_dtype
from flax.linen import module
from flax.linen.module import Module, compact
from flax.typing import (
  Array,
  PRNGKey as PRNGKey,
  Dtype,
  Shape as Shape,
  Initializer,
  PrecisionLike,
  DotGeneralT,
  ConvGeneralDilatedT,
  PaddingLike,
  LaxPadding,
)



class DenseSVD(Module):
  """A linear transformation applied over the last dimension of the input.

  Example usage::

    >>> import flax.linen as nn
    >>> import jax, jax.numpy as jnp

    >>> layer = nn.Dense(features=4)
    >>> params = layer.init(jax.random.key(0), jnp.ones((1, 3)))
    >>> jax.tree_util.tree_map(jnp.shape, params)
    {'params': {'bias': (4,), 'kernel': (3, 4)}}

  Attributes:
    features: the number of output features.
    use_bias: whether to add a bias to the output (default: True).
    dtype: the dtype of the computation (default: infer from input and params).
    param_dtype: the dtype passed to parameter initializers (default: float32).
    precision: numerical precision of the computation see ``jax.lax.Precision``
      for details.
    kernel_init: initializer function for the weight matrix.
    bias_init: initializer function for the bias.
  """

  features: int
  use_bias: bool = True
  dtype: Dtype | None = None
  param_dtype: Dtype = jnp.float32
  precision: PrecisionLike = None
  kernel_init: Initializer = default_kernel_init
  svd_init : Initializer = initializers.uniform(scale=1)
  bias_init: Initializer = initializers.zeros_init()
  # Deprecated. Will be removed.
  dot_general: DotGeneralT | None = None
  dot_general_cls: Any = None

  @compact
  def __call__(self, inputs: Array) -> Array:
    """Applies a linear transformation to the inputs along the last dimension.

    Args:
      inputs: The nd-array to be transformed.

    Returns:
      The transformed input.
    """
    kernel = self.param(
      'kernel',
      self.kernel_init,
      (jnp.shape(inputs)[-1], self.features),
      self.param_dtype,
    )

    svd = self.param(
      'svd',
      self.svd_init,
      (min(jnp.shape(inputs)[-1], self.features),),
      self.param_dtype,
    )
    svd = jnp.sort(jnp.abs(svd),descending=True)

    if self.use_bias:
      bias = self.param(
        'bias', self.bias_init, (self.features,), self.param_dtype
      )
    else:
      bias = None
    inputs, kernel, bias = promote_dtype(inputs, kernel, bias, dtype=self.dtype)

    if self.dot_general_cls is not None:
      dot_general = self.dot_general_cls()
    elif self.dot_general is not None:
      dot_general = self.dot_general
    else:
      dot_general = jax.lax.dot_general

    u,_,vt = jnp.linalg.svd(kernel,full_matrices=False)

    svd_kernel = u @ jnp.diag(svd) @ vt

    y = dot_general(
      inputs,
      svd_kernel,
      (((inputs.ndim - 1,), (0,)), ((), ())),
      precision=self.precision,
    )
    if bias is not None:
      y += jnp.reshape(bias, (1,) * (y.ndim - 1) + (-1,))
    return y

class FullyConnected(nn.Module):
    num_outputs : int
    activation_fn: any

    @nn.compact
    def __call__(self, x ,train: bool = True):
        x = x.reshape(x.shape[0],-1)

        x = DenseSVD(features=512)(x)
        x = self.activation_fn(x)
        x = DenseSVD(features=256)(x)
        x = self.activation_fn(x)
        x = DenseSVD(features=self.num_outputs)(x)

        return x

@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None,None))
def get_loss_fn_vmapped(params,x,y,apply_fn,loss_svd_scale):
    return get_loss_fn(params,x,y,apply_fn,loss_svd_scale)

def cap_2(w):
    return (jnp.max(w["params"]["DenseSVD_0"]["svd"])*jnp.max(w["params"]["DenseSVD_1"]["svd"])*jnp.max(w["params"]["DenseSVD_2"]["svd"]))**2

def cap_F(w):
    return (jnp.sqrt(jnp.sum(jnp.square(w["params"]["DenseSVD_0"]["svd"])))*jnp.max(w["params"]["DenseSVD_0"]["svd"]))**2 + (jnp.sqrt(jnp.sum(jnp.square(w["params"]["DenseSVD_1"]["svd"])))*jnp.max(w["params"]["DenseSVD_1"]["svd"]))**2 + (jnp.sqrt(jnp.sum(jnp.square(w["params"]["DenseSVD_2"]["svd"])))*jnp.max(w["params"]["DenseSVD_2"]["svd"]))**2

def get_loss_fn(params,x,y,apply_fn,loss_svd_scale):
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
    loss_svd = loss_svd_scale*cap_2(params)*cap_F(params)
    loss = loss_task + loss_svd
    # Compute accuracy
    acc = jnp.mean(prediction.argmax(-1)==y)

    return loss,loss_task,loss_svd,acc


@partial(jax.grad,argnums=0,has_aux=True)
def get_grad_fn(params,x,y,apply_fn,loss_svd_scale):
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
        
    loss,loss_task,loss_svd,acc = get_loss_fn(params,x,y,apply_fn,loss_svd_scale)
    return loss, {"loss" : loss,"loss_task" : loss_task, "loss_svd" : loss_svd, "acc" : acc}

@partial(jax.jit,static_argnums=(4,5))
@partial(jax.vmap,in_axes=(0,0,0,0,None,None,None))
def step_fn(params,opt_params,x,y,apply_fn,optim_update_fn,loss_svd_scale):
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
    grad,aux = get_grad_fn(params,x,y,apply_fn,loss_svd_scale)
    
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


def eval(params,apply_fn,ds,loss_svd_scale):
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
        loss,loss_task,loss_svd,acc = get_loss_fn_vmapped(params,x_test,y_test,apply_fn,loss_svd_scale)
        stats_agg.append((loss,loss_task,loss_svd,acc))
    # Since we perform multiple steps, we have to take the average over all steps performed.
    # The resulting loss/accuracy vector will have shape (n,), where n is settings.num_parallel_exps
    loss = np.asarray(jnp.mean(jnp.stack([e[0] for e in stats_agg],axis=0),axis=0))
    loss_task = np.asarray(jnp.mean(jnp.stack([e[1] for e in stats_agg],axis=0),axis=0))
    loss_svd = np.asarray(jnp.mean(jnp.stack([e[2] for e in stats_agg],axis=0),axis=0))
    acc = np.asarray(jnp.mean(jnp.stack([e[3] for e in stats_agg],axis=0),axis=0))

    return loss,loss_task,loss_svd,acc

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

    stats_ckpts = {"train_acc" : {}, "test_acc" : {}, "train_loss" : {}, "test_loss" : {}, "train_loss_task" : {}, "train_loss_svd" : {}, "test_loss_task" : {}, "test_loss_svd" : {}}

    # Initialize the model and ensure that each model is initialized with a different random seed
    model = FullyConnected(10,nn.relu)
    split_key = jax.random.key(random.randint(1,2473673438))
    # Set seed for pythons random for determining time steps where the model is saved and evaluated
    random.seed(42)
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
    for i,(x_train,y_train)in zip(tqdm(range(settings.steps+1)),ds_stack_iterator(*ds_train_train)):

        # Save dense params
        if settings.save_model_every and random.randint(a=1,b=settings.save_model_every) == 1:
            with open(os.path.join(save_path,"states","model",str(i)+".pkl"), "wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,params),f)

        # Save optimizer params
        if settings.save_optim_every and random.randint(a=1,b=settings.save_optim_every) == 1:
            with open(os.path.join(save_path,"states","optim",str(i)+".pkl"), "wb") as f:
                pkl.dump(tree_map(lambda x : np.asarray(x) ,opt_params),f)

        # Perform the gradient update step
        params,opt_params,_ = step_fn(params,opt_params,x_train,y_train,model.apply,optim.update,settings.loss_svd_scale)        
        
        if settings.eval_every and random.randint(a=1,b=settings.eval_every) == 1:

            train_loss,train_loss_task,train_loss_svd,train_acc = eval(params,model.apply,ds_train_eval,settings.loss_svd_scale)
            test_loss,test_loss_task,test_loss_svd,test_acc = eval(params,model.apply,ds_test_eval,settings.loss_svd_scale)

            stats_ckpts["train_loss"][i] = train_loss
            stats_ckpts["train_loss_task"][i] = train_loss_task
            stats_ckpts["train_loss_svd"][i] = train_loss_svd
            stats_ckpts["train_acc"][i] = train_acc

            stats_ckpts["test_loss"][i] = test_loss
            stats_ckpts["test_loss_task"][i] = test_loss_task
            stats_ckpts["test_loss_svd"][i] = test_loss_svd
            stats_ckpts["test_acc"][i] = test_acc

            print("Setting: ", settings.loss_svd_scale, "| step: ",i, " [train_acc: ", train_acc, "| test_acc: ",test_acc, " | train_loss_task: ", train_loss_task, " | train_loss_svd: ", train_loss_svd, "]")
            # Save stats (train/test loss/accuracy)
            with open(os.path.join(save_path,"stats.pkl"),"wb") as f:
                pkl.dump(stats_ckpts,f)
