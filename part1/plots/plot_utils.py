import sys
sys.path.append("/home/miri/Documents/bachelorthesis/part1")
import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from functools import reduce,partial
from itertools import chain
from jax.tree_util import tree_map_with_path,tree_map,tree_transpose,tree_structure,keystr,tree_leaves
jax.config.update('jax_platform_name', 'cpu')
import pickle as pkl
from collections import OrderedDict
import os
import operator
from multiprocessing.pool import ThreadPool
import multiprocessing
from tqdm import tqdm
import pandas as pd
from utils import *

def get_ckpt_paths(path, subfolder):
    if os.path.exists(os.path.join(path,"settings.json")):

        path = os.path.join(path,subfolder)
        files = [f for f in os.listdir(path)]
        files = sorted(files, key = lambda f : int(f.split(".")[0]))

        return OrderedDict([(int(f.split(".")[0]),os.path.join(path,f)) for f in files])
    else:
        dicts = [get_ckpt_paths(os.path.join(path,run),subfolder) for run in os.listdir(os.path.join(path))]
        return tree_map(lambda *e : [*e], *dicts)


def load(path):

    if not type(path) == list:
        path = [path]

    ckpt = []
    for p in path:
        with open(p, "rb") as f:
            ckpt.append(pkl.load(f))
    
    return tree_map(lambda *e : np.concatenate([*e]), *ckpt)

def get_stats(path, subfolder):

    stats = {i : load(p) for i,p in get_ckpt_paths(path, subfolder).items()}
    acc = {key : value["acc"] for key,value in stats.items()}
    loss = {key : value["loss"] for key,value in stats.items()}

    return {"acc" : acc, "loss" : loss}

def plot_step_stat(stats, ax, label=None):

    x = stats.keys()
    mean = np.asarray(list(tree_map(lambda y : np.mean(y),stats).values()))
    std = np.asarray(list(tree_map(lambda y : np.std(y),stats).values()))
    ax.plot(x, mean, label=label)
    ax.fill_between(x, mean-std, mean+std, alpha=0.3)

def get_subexpspaths(path, skip=None):
    for subpath in os.listdir(path):
        if skip:
            if skip(subpath):
                continue
        yield os.path.join(path,subpath),subpath

def max_acc(stats):
    return np.max(np.stack(list(stats["acc"].values())),axis=0)

def plot_hyperparam_y(paths, x_fn, y_fn, ax, label=None, norm=True):
    x = []
    means = []
    stds = []
    for path,subpath in paths:
        y = y_fn(path)
        mean = np.mean(y)
        std = np.std(y)

        x.append(x_fn(subpath))
        means.append(mean)
        stds.append(std)
    
    x,means,stds = np.asarray(x),np.asarray(means),np.asarray(stds)
    argsort_indices = np.argsort(x).astype(np.int32)
    x,means,std = x[argsort_indices],means[argsort_indices],stds[argsort_indices]

    if norm:
        means_normed = (means-np.min(means))/(np.max(means)-np.min(means))
        stds_normed = (stds-np.min(stds))/(np.max(stds)-np.min(stds))
    else:
        means_normed = means
        stds_normed = stds

    ax.plot(x,means_normed, label=label)
    ax.fill_between(x, means_normed-stds_normed, means_normed+stds_normed, alpha=0.3)

    return (x,means,means_normed)


def estimate_convergence(stats, path_length = 5, acc_tol = 0.005, var_tol = 0.5):
    x = np.asarray(list(stats.keys()))
    y = np.asarray(list(stats.values()))
    if len(y.shape) == 1:
        y = np.expand_dims(y,axis=-1)

    step_at_max_acc = np.asarray([x[i] for i in np.where(y-y.max(axis=0)+acc_tol>0,1,0).argmax(axis=0)])
    
    pad = int(path_length/2)

    local_max = jnp.asarray([np.max(y[i-pad:i+pad],axis=0) for i in range(pad,y.shape[0]-pad)])
    local_max = local_max - y.max(axis=0) + acc_tol
    local_max = np.where(local_max>=0,1,0)

    local_var = jnp.asarray([np.var(y[i-pad:i+pad],axis=0) for i in range(pad,y.shape[0]-pad)])
    local_var = local_var - np.min(local_var,axis=0) - var_tol
    local_var = np.where(local_var<=0,1,0)

    convergence = local_max + local_var 

    step = []
    for i in range(convergence.shape[-1]):
        if np.max(convergence[:,i]) < 2:
            step.append(x[-1])
        else:
            step.append(x[np.argmax(convergence[:,i])])

    return step_at_max_acc,np.asarray(step)

    """
    @jax.jit
    @partial(jax.vmap, in_axes=(0,None))
    @partial(jax.vmap, in_axes=(None,0))
    def var_matrix(a,b):
        return jnp.abs((a-b))
    
    converged = []
    for i in range(variances.shape[0]):
        var_m = var_matrix(variances[i:],variances[i:])
        converged.append(jax.vmap(lambda e : jnp.all(e <= var_tol),in_axes=(-1))(var_m))
    converged = jnp.stack(converged).astype(jnp.int32)

    step_at_convergence = np.asarray([x[i] for i in converged.argmax(axis=0)])

    #acc_at_step_at_max_acc = y.max(axis=0)
    #acc_at_step_at_convergence = 
    return step_at_max_acc,step_at_convergence
    """