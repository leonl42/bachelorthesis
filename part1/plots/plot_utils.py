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
    subpaths = []
    for subpath in os.listdir(path):
        if skip:
            if skip(subpath):
                continue
        subpaths.append((os.path.join(path,subpath),subpath))

    return subpaths


def max_acc(stats):
    return np.max(np.stack(list(stats["acc"].values())),axis=0)

def plot_hyperparam_y(paths, x_fn, y_fn, ax, label=None, norm=True):
    if len(paths) == 0:
        return None,None,None
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
    
    if len(x) == 0:
        return

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




def get_batchstats_stats(x, batch_stats):
    batch_stats = tree_map(lambda x : (np.asarray(jnp.mean(x)),np.asarray(jnp.std(x))),batch_stats)
    
    mean_of_mean = {key : {"x" : x, "y" : value["mean"][0]} for key,value in batch_stats.items()}
    var_of_mean = {key : {"x" : x, "y" : value["mean"][1]} for key,value in batch_stats.items()}
    mean_of_var = {key : {"x" : x, "y" : value["var"][0]} for key,value in batch_stats.items()}
    var_of_var = {key : {"x" : x, "y" : value["var"][1]} for key,value in batch_stats.items()}

    return mean_of_mean,var_of_mean,mean_of_var,var_of_var

def get_weights_channel_norms(x, weights):
    weights = {key : value for key,value in weights.items() if "Conv" in key}
    cnorms = tree_map(lambda w : np.asarray(jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0)),weights)
    mean_and_var_of_cnorms = tree_map(lambda x : (np.mean(x),np.var(x)),cnorms)

    cnorm_mean = {key : {"x" : x, "y" : value["kernel"][0]} for key,value in mean_and_var_of_cnorms.items()}
    cnorm_var = {key : {"x" : x, "y" : value["kernel"][1]} for key,value in mean_and_var_of_cnorms.items()}

    return cnorm_mean,cnorm_var

def get_weights_channel_means(x, weights):
    weights = {key : value for key,value in weights.items() if "Conv" in key}
    cmeans = tree_map(lambda w : np.asarray(jnp.mean(w.reshape(-1,w.shape[-1]),axis=0)),weights)
    mean_and_var_of_cmeans = tree_map(lambda x : (np.mean(x),np.var(x)),cmeans)

    cnorm_mean = {key : {"x" : x, "y" : value["kernel"][0]} for key,value in mean_and_var_of_cmeans.items()}
    cnorm_var = {key : {"x" : x, "y" : value["kernel"][1]} for key,value in mean_and_var_of_cmeans.items()}

    return cnorm_mean,cnorm_var

def get_conv_mean_std(x, weights):
    weights = {key : value for key,value in weights.items() if "Conv" in key}
    means_and_std = tree_map(lambda w : (np.asarray(jnp.mean(w)),np.asarray(jnp.std(w))),weights)

    means = {key : {"x" : x, "y" : value["kernel"][0]} for key,value in means_and_std.items()}
    std = {key : {"x" : x, "y" : value["kernel"][1]} for key,value in means_and_std.items()}

    return means,std

def get_conv_norm(x, weights):
    weights = {key : value for key,value in weights.items() if "Conv" in key}
    norms = tree_map(lambda w : np.asarray(jnp.linalg.vector_norm(jnp.reshape(w,-1))),weights)

    norms = {key : {"x" : x, "y" : value["kernel"]} for key,value in norms.items()}

    return norms

def get_optim_momentum_norm(x, optim_state):
    if not hasattr(optim_state[1][0],"trace"):
        return None
    optim_state = optim_state[1][0].trace
    momentum_norm = tree_map(lambda g : jnp.linalg.norm(g.reshape(-1)),optim_state)
    
    momentum_norm_bn = np.mean(np.asarray(tree_leaves({key : value for key,value in momentum_norm.items() if "BatchNorm" in key})))
    momentum_norm_conv = np.mean(np.asarray(tree_leaves({key : value for key,value in momentum_norm.items() if "Conv" in key})))
    momentum_norm_dense = np.mean(np.asarray(tree_leaves({key : value for key,value in momentum_norm.items() if "out" in key})))

    return {"bn" : {"x" : x, "y" : momentum_norm_bn},"conv" : {"x" : x, "y" : momentum_norm_conv},"dense" : {"x" : x, "y" : momentum_norm_dense}}


def plot_data(path, map_fn,start=0,stop=-1, hit = -1, axs_tuple = None):
    # Format of dicts in list:
    # rows
    #   cols
    #       title
    #       label
    #            x
    #            y
    row_dict = []

    ckpt_paths_states = get_ckpt_paths(path,"states")
    ckpt_paths_grads = get_ckpt_paths(path,"grads")

    def compute(i,ckpt_states,_,ckpt_grads):

        if i<start:
            return
        
        if stop != -1 and i>stop:
            return
        
        if hit != -1 and not i%hit == 0:
            return
        
        weights,batch_stats,optim_state = load([ckpt_states])
        grad = load([ckpt_grads])
        
        batchstats_mean_of_mean,batchstats_var_of_mean,batchstats_mean_of_var,batchstats_var_of_var = get_batchstats_stats(i, batch_stats)
        cnorm_mean,cnorm_var = get_weights_channel_norms(i,weights)
        cmean_mean,cmean_var = get_weights_channel_means(i,weights)
        mean,std = get_conv_mean_std(i,weights)
        norm = get_conv_norm(i,weights)
        grad_norm = get_conv_norm(i, grad)
        grad_cnorm_mean,grad_cnorm_var =  get_weights_channel_norms(i,grad)
        grad_mean,grad_std = get_conv_mean_std(i, grad)
        momentumnorm = get_optim_momentum_norm(i,optim_state)
        return map_fn({"batchstats_mean_of_mean" : batchstats_mean_of_mean, "batchstats_var_of_mean" : batchstats_var_of_mean,
                        "batchstats_mean_of_var" : batchstats_mean_of_var, "batchstats_var_of_var" : batchstats_var_of_var,
                       "cnorm_mean" : cnorm_mean,"cnorm_var" : cnorm_var,
                       "cmean_mean" : cmean_mean,"cmean_var" : cmean_var,
                       "mean" : mean,"std" : std,"norm" : norm,
                       "grad_mean" : grad_mean,"grad_std" : grad_std,"grad_norm" : grad_norm, "grad_cnorm_mean" : grad_cnorm_mean,"grad_cnorm_var" : grad_cnorm_var,
                       "momentumnorm" : momentumnorm})
    
 
    with ThreadPool(processes=12) as pool:
        row_dict = pool.map(lambda x : compute(*list(chain(*x))),zip(list(ckpt_paths_states.items()),list(ckpt_paths_grads.items())))
        

    row_dict = [e for e in row_dict if e is not None]
    row_dict = tree_map(lambda *x : x[0] if isinstance(x[0],str) else np.asarray(x) , *row_dict)

    if axs_tuple is None:
        ncols = len(list(row_dict[0].keys()))
        nrows = len(list(row_dict.keys()))
        fig,axs = plt.subplots(ncols = ncols, nrows = nrows)
    else:
        fig,axs,ncols,nrows = axs_tuple

    for row,col_dict in row_dict.items():
        for col, (title, label_dict) in col_dict.items():
            if ncols == 1 and nrows == 1:
                ax = plt
            elif ncols == 1:
                ax = axs[row]
            elif nrows == 1:
                ax = axs[col] 
            else:
                ax = axs[row][col]
            for label,x_y_dict in label_dict.items():
                ax.plot(x_y_dict["x"],x_y_dict["y"],label=label)
            
            ax.set_title(title)
            #ax.legend()

    return (fig,axs)



