import sys
sys.path.append("global_path/code")
import os
import matplotlib.pyplot as plt
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
from itertools import chain
from jax.tree_util import tree_map,tree_leaves
jax.config.update('jax_platform_name', 'cpu')
import pickle as pkl
from collections import OrderedDict
import os
import math
from multiprocessing.pool import ThreadPool
import seaborn as sns
import json
from utils import *




def get_ckpt_paths(path, subfolder):

    path = os.path.join(path,subfolder)
    files = [f for f in os.listdir(path)]
    files = sorted(files, key = lambda f : int(f.split(".")[0]))

    return OrderedDict([(int(f.split(".")[0]),os.path.join(path,f)) for f in files])


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

def plot_hyperparam_y(paths, x_fn, y_fn, ax, label=None, norm=True,alpha=0.3,color=None,plot=True):
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
    
    if plot:
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


def distribution_drift(data_path,exps,labels,colors,drift_keys = ["Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7","out"]):
    fig,axs = plt.subplots(ncols=1,nrows=2,sharey="row")
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{data_path}/{exp}"
        ckpt_paths = get_ckpt_paths(path,"drift")

        x = [e[0] for e in list(ckpt_paths.items())]


        drift = [load(e) for e in ckpt_paths.values()]
        drift = tree_map(lambda *x : jnp.asarray(x) , *[e for e in drift if e is not None])
        drift = tree_map(lambda *x : jnp.asarray(x) , *[drift[key] for key in drift_keys])
    
        for k,measure in enumerate(["dist","cos"]):
            drift_measure = np.mean(drift[measure],axis=0)
            mean = np.mean(drift_measure,axis=-1)
            std = np.std(drift_measure,axis=-1)
            mean,std = smooth(mean,std,smoothing=5)
            axs[k].plot(x,mean,c=color,label=label)
            axs[k].fill_between(x,mean-std,mean+std,alpha=0.15,color=color)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    axs[0].set_ylabel(r"$||\cdot||_2$")
    axs[1].set_ylabel(r"$\cos(\theta)$")
    
    return fig,axs


def adam_drift(data_path,exps,labels,colors,drift_keys = ["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]):
    fig,axs = plt.subplots(ncols=1,nrows=2,sharey="row")
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{data_path}/{exp}"
        ckpt_paths = get_ckpt_paths(path,"adam_drift")

        x = [e[0] for e in list(ckpt_paths.items())]


        drift = [load(e) for e in ckpt_paths.values()]
        drift = tree_map(lambda *x : jnp.asarray(x) , *[e for e in drift if e is not None])
        drift = tree_map(lambda x : jnp.mean(x,axis=-1),drift)
        drift = tree_map(lambda *x : jnp.asarray(x) , *[drift[key] for key in drift_keys])
        
        for k,measure in enumerate(["nu","mu"]):
            drift_measure = np.mean(drift[measure]["cos"],axis=0)
            mean = np.mean(drift_measure,axis=-1)
            std = np.std(drift_measure,axis=-1)
            mean,std = smooth(mean,std,smoothing=5)
            axs[k].plot(x,mean,c=color,label=label)
            axs[k].fill_between(x,mean-std,mean+std,alpha=0.15,color=color)

    lines, labels = axs[0].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    axs[0].set_title("Cosine Similarity of Adam running average $\mu_{adam}$ and calculated $\mu_{calc}$",font={'weight' : 'normal'})
    axs[1].set_title("Cosine Similarity of Adam running average $\\nu_{adam}$ and calculated $\\nu_{calc}$",font={'weight' : 'normal'})

    axs[0].set_ylabel(r"$\cos(\theta)$")
    axs[1].set_ylabel(r"$\cos(\theta)$")

    return fig,axs

def sgdm_drift(data_path,exps,labels,colors,drift_keys = ["Conv_0","Conv_1","Conv_2","Conv_3","Conv_4","Conv_5","Conv_6","Conv_7"]):
    fig,axs = plt.gcf(),plt.gca()
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{data_path}/{exp}"
        ckpt_paths = get_ckpt_paths(path,"sgdm_drift")

        x = [e[0] for e in list(ckpt_paths.items())]


        drift = [load(e) for e in ckpt_paths.values()]
        drift = tree_map(lambda *x : jnp.asarray(x) , *[e for e in drift if e is not None])
        drift = tree_map(lambda x : jnp.mean(x,axis=-1),drift)
        drift = tree_map(lambda *x : jnp.asarray(x) , *[drift[key] for key in drift_keys])

        drift_measure = np.mean(drift["trace"]["cos"],axis=0)

        mean = np.mean(drift_measure,axis=-1)
        std = np.std(drift_measure,axis=-1)
        mean,std = smooth(mean,std,smoothing=5)
        axs.plot(x,mean,c=color,label=label)
        axs.fill_between(x,mean-std,mean+std,alpha=0.15,color=color)

    lines, labels = axs.get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    axs.set_title("Cosine Similarity of SgdM running average $\mu_{sgdm}$ and calculated $\mu_{calc}$",font={'weight' : 'normal'})

    axs.set_ylabel(r"$\cos(\theta)$")


    return fig,axs


def plot_mean_or_norm(exps,labels,colors,mg_spacing,load_idx=0,plot_mean=True,max_step=None,layer="conv",attr="kernel",measure_global=False,sharey="all"):
    
    fig,axs = plt.subplots(ncols=len(labels),nrows=1,sharey=sharey)
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{exp}/{mg_spacing}"
        ckpt_paths = get_ckpt_paths(path,"states")
        ckpt_paths[0] = f"{exp}/states/0.pkl"
        ckpt_paths.move_to_end(0, last=False)
        
        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_mean_fn(weights):

            if plot_mean:
                measure_fn = lambda w : jnp.mean(w,axis=0)
            else:
                measure_fn = lambda w : jnp.linalg.vector_norm(w,axis=0)

            if measure_global:
                measure = tree_map(lambda w : measure_fn(w.reshape(-1)),weights)
            else:
                measure = tree_map(lambda w : jnp.mean(measure_fn(jnp.reshape(w,(-1,w.shape[-1])))),weights)

            return measure
        
        x = [e for e in list(ckpt_paths.keys())]
        with ThreadPool(processes=12) as pool:
            measures = pool.map(lambda e : get_mean_fn(load(e)[load_idx]),ckpt_paths.values())

        measures = tree_map(lambda *x : jnp.asarray(x) , *[e for e in measures if e is not None])
        measures = {key : val for key,val in measures.items() if  substrings_in_path(key.lower(),layer)}
        
        #(layer, t, num_runs)
        measures = jnp.asarray([val[attr] for _,val in measures.items()])
        mean = np.mean(measures,axis=(0,-1))
        std = np.std(measures,axis=(0,-1))

        if max_step:
            cutoff = len([e for e in x if e<=max_step])
            x = x[:cutoff]
            mean = mean[:cutoff]
            std = std[:cutoff]
        axs[i].plot(x,mean,c=color)
        axs[i].fill_between(x,mean-std,mean+std,alpha=.15,color=color)

        axs[i].set_title(label,font={'weight' : 'normal'})
    
    return fig,axs


def gradients_and_updates(data_path,exps,labels,colors,mg_spacing,layer="conv"):
    
    fig,axs = plt.subplots(ncols=1,nrows=2)
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
        path = f"{data_path}/{exp}/{mg_spacing}"
        ckpt_paths_grads = get_ckpt_paths(path,"grads")
        ckpt_paths_updates = get_ckpt_paths(path,"updates")

        @jax.jit
        @partial(jax.vmap,in_axes=(0))
        def get_grad_norm_fn(w):
            norms = tree_map(lambda w : jnp.mean(jnp.linalg.vector_norm(jnp.reshape(w,(-1,w.shape[-1])),axis=0),axis=0),w)
            return norms


        x = [e[0] for e in list(ckpt_paths_updates.items())]

        with ThreadPool(processes=12) as pool:
            grads = pool.map(lambda e : get_grad_norm_fn(load(e)),ckpt_paths_grads.values())
            updates = pool.map(lambda e : get_grad_norm_fn(load(e)),ckpt_paths_updates.values())
        
        def to_jnp(row_dict):
            row_dict = tree_map(lambda *x : jnp.asarray(x) , *[e for e in row_dict if e is not None])
            row_dict = {key : val for key,val in row_dict.items() if substrings_in_path(key.lower(),layer)}
            row_dict = jnp.asarray([val["kernel"] for _,val in row_dict.items()]).T
        
            return row_dict
        
        grads,updates = to_jnp(grads),to_jnp(updates)

        grads = jnp.mean(grads,axis=-1)
        updates = jnp.mean(updates,axis=-1)
        

        def plot(i,polyfit):
            mean = np.mean(polyfit,axis=0)
            std = np.std(polyfit,axis=0)
            mean,std = smooth(mean,std,4)
            axs[i].plot(x,mean,label=label,c=color)
            axs[i].fill_between(x, mean-std, mean+std, alpha=0.15,color=color)

        plot(0,grads)
        plot(1,updates)

    lines, labels = axs[1].get_legend_handles_labels()
    fig.legend(lines, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5,-0.075*math.ceil(len(labels)/2)), bbox_transform=fig.transFigure)

    axs[0].set_title("Norm of gradient of output channel weights",font={'weight' : 'normal'})
    axs[1].set_title("Norm of updates of output channel weights",font={'weight' : 'normal'})

    axs[0].set_ylabel(r"$||\cdot||_2$",font={'weight' : 'normal'},labelpad=10)
    axs[1].set_ylabel(r"$||\cdot||_2$",font={'weight' : 'normal'},labelpad=10)

    return fig,axs


def mg_spacing(data_path,exps,labels,colors,mg_spacing,layer="conv"):
    
    fig,axs = plt.subplots(ncols=1,nrows=3)
    for i,(label,exp,color) in enumerate(zip(labels,exps,colors)):
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
            row_dict = {key : val for key,val in row_dict.items() if substrings_in_path(key.lower(),layer)}
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


def get_hyperparam_best_acc(data_path,exps,hyperparam_fns):
     for exp,hyperparam_fn in zip(exps,hyperparam_fns):
        x,y,_ = plot_hyperparam_y(get_subexpspaths(f"{data_path}/{exp}"),
                                hyperparam_fn, 
                                lambda p : max_acc(get_stats(p,"test_stats")),
                                plt, 
                                norm=False,
                                plot=False)
        print(f"Best of {exp} is {round(x[y.argmax()],8)} & {round(100*y.max(),2)}\\%")

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

def plot_wbn_best_hyperparameter_validation_curve(data_path,image_path,exps,labels,colors,split,max_step):
    fig,axs = plt.subplots(ncols=2,nrows=1,sharey="row")

    for exp,label,color in zip(exps[:split],labels[:split],colors[:split]):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],axs[0],label=label,smoothing=5,color=color,alpha=0.15,max_step=max_step)

    for exp,label,color in zip(exps[split:],labels[split:],colors[split:]):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],axs[1],label=label,smoothing=5,color=color,alpha=0.15,max_step=max_step)

    axs[0].set_ylim(0.7,0.9)
    axs[0].legend()
    axs[1].legend()
    fig.text(-0.03, 0.5, "Validation accuracy", va='center', rotation='vertical',font={'size'   : 12,'weight' : 'bold'})
    fig.set_size_inches(12,6)
    fig.tight_layout()
    fig.savefig(f"{image_path}/best_hyperparameter_validation_accuracy.png", bbox_inches='tight')


def plot_wobn_best_hyperparameter_validation_curve(data_path,image_path,exps,labels,colors,max_step):

    for exp,label,color in zip(exps,labels,colors):
        plot_step_stat(get_stats(f"{data_path}/{exp}","test_stats")["acc"],plt,label=label,smoothing=5,color=color,alpha=0.15,max_step=max_step)

    plt.gca().set_ylim(0.7,0.9)
    plt.legend()

    plt.gcf().text(-0.03, 0.5, "Validation accuracy", va='center', rotation='vertical',font={'size'   : 12,'weight' : 'normal'})
    
    return plt.gcf(),plt.gca()