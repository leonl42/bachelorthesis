import sys
sys.path.append("/home/miri/Documents/bachelorthesis/code")
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
import math
import operator
from multiprocessing.pool import ThreadPool
import seaborn as sns
import multiprocessing
from tqdm import tqdm
import pandas as pd
import json
from utils import *


class write:
    def __init__(self,name,path,h,max_tasks):
        self.i = 1
        self.k = 0
        self.name = name
        self.path = path
        self.h = h
        self.max_tasks=max_tasks

        os.makedirs(self.path,exist_ok=True)

        with open(f"{self.path}/run.sh","w") as f:
            pass

    def create_file(self):
        with open(f"{self.path}/run_{self.i}.sbatch","w") as f:
            f.write(f"""#!/bin/bash
#SBATCH --job-name="{self.name}_{self.i}"
#SBATCH --cpus-per-task=16            
#SBATCH --mem=32G                     
#SBATCH --time={self.h}:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1 
#SBATCH --constraint="A100|H100.80gb"
#SBATCH --output={self.name}_{self.i}.out

source ~/miniconda/bin/activate deskitopi

export TMPDIR=/share/users/student/l/llemke/tmp
export PATH=/home/student/l/llemke/miniconda/envs/deskitopi/bin:/home/student/l/llemke/miniconda/condabin:/opt/xcat/bin:/opt/xcat/sbin:/opt/xcat/share/xcat/tools:/appl/spack/bin:/usr/lib64/ccache

""")
        with open(f"{self.path}/run.sh","a") as f:
            f.write(f"sbatch {self.path}/run_{self.i}.sbatch \n")


        self.i += 1


    def write(self, s):
        if self.k == 0:
            self.create_file()
        
        with open(f"{self.path}/run_{self.i-1}.sbatch","a") as f:
            f.write(s)

        self.k += 1

        if self.k >= self.max_tasks:
            self.k = 0



def get_ckpt_paths(path, subfolder):
    #if os.path.exists(os.path.join(path,"settings.json")):

    path = os.path.join(path,subfolder)
    files = [f for f in os.listdir(path)]
    files = sorted(files, key = lambda f : int(f.split(".")[0]))

    return OrderedDict([(int(f.split(".")[0]),os.path.join(path,f)) for f in files])
    #else:
    #    dicts = [get_ckpt_paths(os.path.join(path,run),subfolder) for run in os.listdir(os.path.join(path))]
    #    return tree_map(lambda *e : [*e], *dicts)


def load(path):

    if not type(path) == list:
        path = [path]

    ckpt = []
    for p in path:
        with open(p, "rb") as f:
            ckpt.append(pkl.load(f))
            
    return tree_map(lambda *e : np.concatenate([*e]), *ckpt)

def get_stats(path, subfolder):

    if not os.path.exists(os.path.join(path,subfolder)):
        return None

    stats = {}
    ckpt_paths = list(get_ckpt_paths(path, subfolder).items())

    if len(ckpt_paths) == 0:
        return None

    for i,p in ckpt_paths:
        if os.path.getsize(p) != 0:
            stats[i] = load(p)
        else:
            print(f"{'\033[91m'} File: {p} had size {os.path.getsize(p)} {'\033[91m'}")
    
    acc = {key : value["acc"] for key,value in stats.items()}
    loss = {key : value["loss"] for key,value in stats.items()}

    return {"acc" : acc, "loss" : loss}

def smooth(mean,std,smoothing):
    new_mean = np.zeros_like(mean)
    new_std = np.zeros_like(std)
    for i in range(mean.size):
        new_mean[i] = np.mean(mean[max(i-smoothing,0):min(i+smoothing+1,mean.size)])
        new_std[i] = np.mean(std[max(i-smoothing,0):min(i+smoothing+1,mean.size)])
    return new_mean,new_std


def plot_step_stat(stats, ax, label=None, smoothing : int = 0, show_std = True,alpha=0.3,max_step=None,color=None):

    x = list(stats.keys())



    mean = list(tree_map(lambda y : np.mean(y),stats).values())
    std = list(tree_map(lambda y : np.std(y),stats).values())

    if x[0] != 0:
        x.insert(0,0)
        mean.insert(0,0.1)
        std.insert(0,0)

    mean = np.asarray(mean)
    std = np.asarray(std)
    
    if max_step:
        new_x = [e for e in x if e<=max_step]
        mean = mean[:len(new_x)]
        std = std[:len(new_x)]
        x = new_x

    if smoothing > 0:
        mean,std = smooth(mean,std,smoothing)
    if color:
        ax.plot(x, mean, label=label,c=color)
        if show_std:
            ax.fill_between(x, mean-std, mean+std, alpha=alpha,color=color)
    else:
        ax.plot(x, mean, label=label)
        if show_std:
            ax.fill_between(x, mean-std, mean+std, alpha=alpha)

def get_subexpspaths(path, skip=None):
    subpaths = []
    for subpath in os.listdir(path):
        if skip:
            if skip(subpath):
                continue
        subpaths.append(os.path.join(path,subpath))

    return subpaths


def max_acc(stats):
    if stats is None:
        return None
    
    return np.max(np.stack(list(stats["acc"].values())),axis=0)

def plot_hyperparam_y(paths, x_fn, y_fn, ax, label=None, norm=True,alpha=0.3,color=None):
    if len(paths) == 0:
        return None,None,None
    x = []
    means = []
    stds = []
    for path in paths:
        y = y_fn(path)
        if y is None:
            continue
        mean = np.mean(y)
        std = np.std(y)

        with open(os.path.join(path,"settings.json"),"r") as f:
            x.append(x_fn(json.load(f)))
        means.append(mean)
        stds.append(std)
    
    if len(x) == 0:
        return None, None, None

    x,means,stds = np.asarray(x),np.asarray(means),np.asarray(stds)
    argsort_indices = np.argsort(x).astype(np.int32)
    x,means,std = x[argsort_indices],means[argsort_indices],stds[argsort_indices]

    if norm:
        means_normed = (means-np.min(means))/(np.max(means)-np.min(means))
        stds_normed = (stds-np.min(stds))/(np.max(stds)-np.min(stds))
    else:
        means_normed = means
        stds_normed = stds

    if color:
        ax.plot(x,means_normed, label=label,c=color)
        ax.fill_between(x, means_normed-stds_normed, means_normed+stds_normed, alpha=alpha,color=color)
    else:
        ax.plot(x,means_normed, label=label)
        ax.fill_between(x, means_normed-stds_normed, means_normed+stds_normed, alpha=alpha)


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
    try:
        mean_of_mean = {key : {"x" : x, "y" : value["mean"][0]} for key,value in batch_stats.items()}
        var_of_mean = {key : {"x" : x, "y" : value["mean"][1]} for key,value in batch_stats.items()}
        mean_of_var = {key : {"x" : x, "y" : value["var"][0]} for key,value in batch_stats.items()}
        var_of_var = {key : {"x" : x, "y" : value["var"][1]} for key,value in batch_stats.items()}

        return mean_of_mean,var_of_mean,mean_of_var,var_of_var
    except:
        return {},{},{},{}
    
def get_weights_channel_norms(x, weights):
    weights = {key : value for key,value in weights.items() if "Conv" in key}
    cnorms = tree_map(lambda w : jnp.linalg.vector_norm(w.reshape(-1,w.shape[-1]),axis=0,keepdims=False),weights)
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


def plot_data(path, map_fn,start=0,stop=-1, hit = -1, axs_tuple = None,states="states", return_row_dict=False,plot=True):
    # Format of dicts in list:
    # rows
    #   cols
    #       title
    #       label
    #            x
    #            y
    row_dict = []

    ckpt_paths_states = get_ckpt_paths(path,states)
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
        
        weights = tree_map(lambda w : w[0], weights)
        batch_stats = tree_map(lambda w : w[0], batch_stats)
        optim_state = tree_map(lambda w : w[0], optim_state)
        grad = tree_map(lambda w : w[0], grad)

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
    
    if plot:
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
    else:
        fig,axs = None,None
    if return_row_dict:
        return (fig,axs),row_dict
    else:
        return (fig,axs)



def mg_spacing2(data_path,exps,labels,colors,mg_spacing,ncols,nrows,smoothing):
    fig,axs = plt.subplots(ncols=ncols,nrows=nrows,sharey="all")
    axs = axs.flatten()
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors.values())):
        path = f"{data_path}/{exp}/{mg_spacing}"
        ckpt_paths_grads = get_ckpt_paths(path,"grads")

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_grad_norm_fn(grads):
            norms = tree_map(lambda w : jnp.linalg.vector_norm(jnp.reshape(w,-1)),grads)
            return norms

        x = [e[0] for e in list(ckpt_paths_grads.items())]
        with ThreadPool(processes=12) as pool:
            row_dict = pool.map(lambda e : get_grad_norm_fn(load([e[1]])),list(ckpt_paths_grads.items()))
        
        row_dict = tree_map(lambda *x : jnp.asarray(x) , *[e for e in row_dict if e is not None])
        row_dict = {key : val for key,val in row_dict.items() if not "batch" in key.lower()}
        row_dict = np.asarray([val["kernel"] for _,val in row_dict.items()])

        
        mean = np.mean(row_dict,axis=-1)
        std = np.std(row_dict,axis=-1)

        

        for l in range(mean.shape[0]):

            plt_mean,plt_std = smooth(mean[l,:],std[l,:],smoothing)

            axs[i].plot(x,plt_mean,label=l)
            axs[i].fill_between(x, plt_mean - plt_std, plt_mean + plt_std,alpha=0.15)

    fig.set_size_inches(6*ncols,6*nrows)
    fig.tight_layout()

    return fig,axs


def lars(data_path,exps,labels,colors,lrs,mg_spacing,folder):
    
    fig,axs = plt.subplots(ncols=len(labels),nrows=1,sharey="all")
    for i,(label,exp,color,lr) in enumerate(zip(labels,exps,colors.values(),lrs)):
        path = f"{data_path}/{exp}/{mg_spacing}"
        ckpt_paths_grads = get_ckpt_paths(path,folder)
        ckpt_paths_states = get_ckpt_paths(path,"states")

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_grad_norm_fn(v):
            norms = tree_map(lambda w : jnp.linalg.vector_norm(jnp.reshape(w,-1)),v)
            return norms
        
        x = [e for e in list(ckpt_paths_grads.keys())]
        with ThreadPool(processes=12) as pool:
            grads = pool.map(lambda e : get_grad_norm_fn(load(e)),ckpt_paths_grads.values())
            states = pool.map(lambda e : get_grad_norm_fn(load(e)[0]),ckpt_paths_states.values())

        grads = tree_map(lambda *x : jnp.asarray(x) , *[e for e in grads if e is not None])
        states = tree_map(lambda *x : jnp.asarray(x) , *[e for e in states if e is not None])

        grads = {key : val for key,val in grads.items() if not "batch" in key.lower()}
        states = {key : val for key,val in states.items() if not "batch" in key.lower()}

        grads = jnp.asarray([val["kernel"] for _,val in grads.items()]).T
        states = jnp.asarray([val["kernel"] for _,val in states.items()]).T

        res = states/grads

        mean = np.mean(res,axis=0)
        std = np.std(res,axis=0)
        for l in range(mean.shape[-1]):
            axs[i].plot(x,mean[:,l])
    
    return fig,axs

def grad_norm_per_layer(data_path,exps,labels,colors,mg_spacing,folder):
    
    fig,axs = plt.subplots(ncols=len(labels),nrows=1,sharey="all")
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors.values())):
        path = f"{data_path}/{exp}/{mg_spacing}"
        ckpt_paths_grads = get_ckpt_paths(path,folder)

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_grad_norm_fn(grads):
            norms = tree_map(lambda w : jnp.linalg.vector_norm(jnp.reshape(w,-1)),grads)
            return norms
        
        x = [e for e in list(ckpt_paths_grads.keys())]
        with ThreadPool(processes=12) as pool:
            grads = pool.map(lambda e : get_grad_norm_fn(load(e)),ckpt_paths_grads.values())

        grads = tree_map(lambda *x : jnp.asarray(x) , *[e for e in grads if e is not None])
        grads = {key : val for key,val in grads.items() if not "batch" in key.lower()}
        grads = jnp.asarray([val["kernel"] for _,val in grads.items()]).T


        mean = np.mean(grads,axis=0)
        std = np.std(grads,axis=0)
        for l in range(mean.shape[-1]):
            axs[i].plot(x,mean[:,l])
    
    return fig,axs

def distribution_drift(data_path,exps,labels,colors):
    fig,axs = plt.subplots(ncols=1,nrows=2,sharey="row")
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{data_path}/{exp}"
        ckpt_paths = get_ckpt_paths(path,"drift")

        x = [e[0] for e in list(ckpt_paths.items())]


        drift = [load(e) for e in ckpt_paths.values()]
        drift = tree_map(lambda *x : jnp.asarray(x) , *[e for e in drift if e is not None])
        drift = tree_map(lambda *x : jnp.asarray(x) , *[drift[key] for key in ["Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7","out"]])
    
        for k,measure in enumerate(["dist","cos"]):
            drift_measure = np.mean(drift[measure],axis=0)
            mean = np.mean(drift_measure,axis=-1)
            std = np.std(drift_measure,axis=-1)
            mean,std = smooth(mean,std,smoothing=5)
            axs[k].plot(x,mean,c=color,label=label)
            axs[k].fill_between(x,mean-std,mean+std,alpha=0.15,color=color)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    return fig,axs

def plot_mean_or_norm(exps,labels,mg_spacing,plot_mean=True,max_step=None):
    
    fig,axs = plt.subplots(ncols=len(labels),nrows=1)
    for i,(label,exp) in enumerate(zip(labels,exps)):
        path = f"{exp}/{mg_spacing}"
        ckpt_paths = get_ckpt_paths(path,"states")

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_mean_fn(weights):
            if plot_mean:
                measure = tree_map(lambda w : jnp.mean(jnp.mean(jnp.reshape(w,(-1,w.shape[-1])),axis=0),axis=0),weights)
            else:
                measure = tree_map(lambda w : jnp.mean(jnp.linalg.vector_norm(jnp.reshape(w,(-1,w.shape[-1])),axis=0),axis=0),weights)
            return measure
        
        x = [e for e in list(ckpt_paths.keys())]
        with ThreadPool(processes=12) as pool:
            measures = pool.map(lambda e : get_mean_fn(load(e)[0]),ckpt_paths.values())

        measures = tree_map(lambda *x : jnp.asarray(x) , *[e for e in measures if e is not None])
        measures = {key : val for key,val in measures.items() if not "batch" in key.lower() or "out" in key.lower()}
        #(layer, t, num_runs)
        measures = jnp.asarray([val["kernel"] for _,val in measures.items()])

        mean = np.mean(measures,axis=(0,-1))
        std = np.std(measures,axis=(0,-1))

        if max_step:
            cutoff = len([e for e in x if e<=max_step])
            x = x[:cutoff]
            mean = mean[:cutoff]
            std = std[:cutoff]
        axs[i].plot(x,mean,c=sns.color_palette("tab10", 1)[0])
        axs[i].fill_between(x,mean-std,mean+std,alpha=.15,color=sns.color_palette("tab10", 1)[0])

        axs[i].set_title(label,font={'weight' : 'bold'})
    
    return fig,axs


def mg_spacing(data_path,exps,labels,colors,mg_spacing):
    
    fig,axs = plt.subplots(ncols=1,nrows=3)
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors.values())):
        path = f"{data_path}/{exp}/{mg_spacing}"
        ckpt_paths_updates = get_ckpt_paths(path,"updates")
        ckpt_paths_states = get_ckpt_paths(path,"states")

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_grad_norm_fn(grads):
            norms = tree_map(lambda w : jnp.linalg.vector_norm(jnp.reshape(w,-1)),grads)
            return norms

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        @partial(jax.vmap,in_axes=(0))
        def polyfit_fn(norms):

            return jnp.polyfit(jnp.arange(norms.size,dtype=jnp.float32),norms,1)

        x = [e[0] for e in list(ckpt_paths_updates.items())]

        with ThreadPool(processes=12) as pool:
            states = pool.map(lambda e : get_grad_norm_fn(load(e)[0]),ckpt_paths_states.values())
            updates = pool.map(lambda e : get_grad_norm_fn(load(e)),ckpt_paths_updates.values())
        
        def to_jnp(row_dict):
            row_dict = tree_map(lambda *x : jnp.asarray(x) , *[e for e in row_dict if e is not None])
            row_dict = {key : val for key,val in row_dict.items() if not "batch" in key.lower()}
            row_dict = jnp.asarray([val["kernel"] for _,val in row_dict.items()]).T
        
            return row_dict
        
        states,updates = to_jnp(states),to_jnp(updates)

        polyfit_states = tree_map(polyfit_fn,states)
        polyfit_updates = tree_map(polyfit_fn,updates)

        def plot(i,polyfit):
            mean = np.mean(polyfit,axis=0)
            std = np.std(polyfit,axis=0)
            mean,std = smooth(mean,std,4)
            axs[i].plot(x,mean,label=label,c=color)
            axs[i].fill_between(x, mean-std, mean+std, alpha=0.15,color=color)

        plot(0,polyfit_states[:,:,1])
        plot(1,polyfit_updates[:,:,0])
        plot(2,polyfit_updates[:,:,1])

    lines, labels = axs[1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    axs[0].set_title("weight b",font={'weight' : 'bold'})
    axs[1].set_title("grad m",font={'weight' : 'bold'})
    axs[2].set_title("grad b",font={'weight' : 'bold'})

    return fig,axs

colors = dict(zip(["noreg","norm","cnorm","cnormu","gcstdu","wd"], sns.color_palette("tab10", 6)))


def plot_wbn_setting_hyperparam_max_acc(data_path,image_path,wrs_p1,wrs):
    fig,axs = plt.subplots(ncols=3,nrows=1,sharey="row")

    x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/noreg"),
                            lambda js : np.sqrt(0.001/(js["optimizer"]["lr"])), 
                            lambda p : max_acc(get_stats(p,"test_stats")),
                            axs[0], 
                            norm=False,
                            label="Noreg",
                            color=colors["noreg"],
                            alpha=0.15)
    print("Max test accuracy of Standard is {0}% with lr {1}".format(round(100*y.max(),2),round(0.001/(x[y.argmax()]**2),6)))

    for i,exp in enumerate(wrs_p1):
        x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/{exp.lower()}"),
                                lambda js: np.sqrt(0.001/(js["optimizer"]["lr"])), 
                                lambda p : max_acc(get_stats(p,"test_stats")), 
                                axs[0], 
                                norm=False, 
                                label=rf"{exp.split("_")[0]}($p=1$)",
                                color=colors[exp.split("_")[0].lower()],
                                alpha=0.15)
        print("Max test accuracy of {0} is {1}% with p {2}".format(exp,round(100*y.max(),2),round(0.001/(x[y.argmax()]**2),6)))


    for i,exp in enumerate(wrs):

        x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/{exp.lower()}"),
                                lambda js: js["norm"]["norm_multiply"],
                                lambda p : max_acc(get_stats(p,"test_stats")), 
                                axs[1], 
                                norm=False, 
                                label=rf"${exp}($lr=0.001$)$",
                                color=colors[exp.lower()],
                                alpha=0.15)
        print("Max test accuracy of {0} is {1}% with p {2}".format(exp,round(100*y.max(),2),x[y.argmax()]))

    x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/wd"),
                            lambda js : np.log(js["optimizer"]["lambda_wd"]),
                            lambda p : max_acc(get_stats(p,"test_stats")),
                            axs[2], 
                            norm=False,
                            label=r"WD($lr=0.001$)",
                            color=colors["wd"],
                            alpha=0.15)
    print("Max test accuracy of wd is {0}% with lambda {1}".format(round(100*y.max(),2),np.exp(x[y.argmax()])))

    axs[0].set_ylim(0.7,0.9)

    axs[0].legend()
    axs[1].legend()
    axs[2].legend()

    fig.text(-0.03, 0.5, "Max. Validation accuracy", va='center', rotation='vertical',font={'size'   : 14,'weight' : 'bold'})

    axs[0].set_xlabel(r"$\mathbf{\sqrt{\frac{0.001}{lr}}}$",font={'size'   : 12,'weight' : 'bold'},labelpad=20)
    axs[1].set_xlabel(r"$\mathbf{p}$",font={'size'   : 12,'weight' : 'bold'},labelpad=20)
    axs[2].set_xlabel(r"$\mathbf{\text{log} \left( \lambda_{wd} \right)}$",font={'size'   : 12,'weight' : 'bold'},labelpad=20)

    fig.set_size_inches(18,6)
    fig.tight_layout()

    fig.savefig(f"{image_path}/setting_hyperparameter_max_accuracy.png", bbox_inches='tight')

def plot_wobn_setting_hyperparam_max_acc(data_path,image_path,wrs,lr):
    fig,axs = plt.subplots(ncols=2,nrows=1,sharey="row")


    for i,exp in enumerate(wrs):

        x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/{exp.lower()}"),
                                lambda js: js["norm"]["norm_multiply"],
                                lambda p : max_acc(get_stats(p,"test_stats")), 
                                axs[0], 
                                norm=False, 
                                label=rf"${exp}($lr={lr}$)$",
                                color=colors[exp.lower()],
                                alpha=0.15)
        print("Max test accuracy of {0} is {1}% with p {2}".format(exp,round(100*y.max(),2),x[y.argmax()]))

    x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/wd"),
                            lambda js : np.log(js["optimizer"]["lambda_wd"]),
                            lambda p : max_acc(get_stats(p,"test_stats")),
                            axs[1], 
                            norm=False,
                            label=rf"WD($lr={lr}$)",
                            color=colors["wd"],
                            alpha=0.15)
    print("Max test accuracy of wd is {0}% with lambda {1}".format(round(100*y.max(),2),np.exp(x[y.argmax()])))

    axs[0].set_ylim(0.7,0.9)

    axs[0].legend()
    axs[1].legend()

    fig.text(-0.03, 0.5, "Max. Validation accuracy", va='center', rotation='vertical',font={'size'   : 14,'weight' : 'bold'})

    axs[0].set_xlabel(r"$\mathbf{p}$",font={'size'   : 10,'weight' : 'bold'},labelpad=20)
    axs[1].set_xlabel(r"$\mathbf{\text{log} \left( \lambda_{wd} \right)}$",font={'size'   : 10,'weight' : 'bold'},labelpad=20)

    fig.set_size_inches(12,6)
    fig.tight_layout()

    fig.savefig(f"{image_path}/setting_hyperparameter_max_accuracy.png", bbox_inches='tight')

def plot_wbn_best_hyperparameter_validation_curve(data_path,image_path,exps,labels,settings,split,max_step):
    fig,axs = plt.subplots(ncols=2,nrows=1,sharey="row")

    for exp,label,setting in zip(exps[:split],labels[:split],settings[:split]):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],axs[0],label=label,smoothing=5,color=colors[setting],alpha=0.15,max_step=max_step)

    for exp,label,setting in zip(exps[split:],labels[split:],settings[split:]):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],axs[1],label=label,smoothing=5,color=colors[setting],alpha=0.15,max_step=max_step)

    axs[0].set_ylim(0.7,0.9)
    axs[0].legend()
    axs[1].legend()
    fig.text(-0.03, 0.5, "Validation accuracy", va='center', rotation='vertical',font={'size'   : 12,'weight' : 'bold'})
    fig.set_size_inches(12,6)
    fig.tight_layout()
    fig.savefig(f"{image_path}/best_hyperparameter_validation_accuracy.png", bbox_inches='tight')


def plot_wobn_best_hyperparameter_validation_curve(data_path,image_path,exps,labels,settings,max_step):

    for exp,label,setting in zip(exps,labels,settings):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],plt,label=label,smoothing=5,color=colors[setting],alpha=0.15,max_step=max_step)

    plt.gca().set_ylim(0.7,0.9)
    plt.legend()

    plt.gcf().text(-0.03, 0.5, "Validation accuracy", va='center', rotation='vertical',font={'size'   : 12,'weight' : 'bold'})
    plt.gcf().set_size_inches(6,6)
    plt.gcf().tight_layout()
    plt.gcf().savefig(f"{image_path}/best_hyperparameter_validation_accuracy.png", bbox_inches='tight')