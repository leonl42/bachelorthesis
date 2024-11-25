import jax
import jax.numpy as jnp
import optax
from functools import partial
from models import VGG11,ResNet,BottleneckResNetBlock
from prepare_dataset import get_cifar
from jax.tree_util import tree_map_with_path,keystr,tree_map,tree_leaves_with_path,tree_leaves
import flax.linen as nn
from types import SimpleNamespace

@partial(jax.jit,static_argnums=(4,))
@partial(jax.vmap,in_axes=(0,0,0,0,None))
def test_step(weights, batch_stats, image, label, apply_fn):
    prediction = apply_fn({"params" : weights, "batch_stats" : batch_stats} ,image, train=False)
    
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(prediction,label))
    acc = jnp.mean((jnp.argmax(prediction,axis=-1) == label)*1)

    return {"loss" : loss, "acc" : acc}

def eval(weights,batch_stats,apply_fn,ds,its):
    aux_agg = []

    for _,(img,lbl) in zip(range(its),ds):
        aux = test_step(weights,batch_stats,img,lbl,apply_fn)
        aux_agg.append(aux)

    return tree_map(lambda *x : jnp.mean(jnp.stack(x,axis=0),axis=0),*aux_agg)


def get_loss_fn(weights, batch_stats, image, label, key, apply_fn):

    prediction,updates = apply_fn({"params" : weights, "batch_stats" : batch_stats}, image, train=True, mutable=["batch_stats"],rngs={"dropout" : key})
    
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(prediction,label))
    acc = jnp.mean((jnp.argmax(prediction,axis=-1) == label)*1)

    return loss,{"loss" : loss, "acc" : acc, "batch_stats" : updates["batch_stats"]}


@partial(jax.jit,static_argnums=5)
@partial(jax.vmap,in_axes=(0,0,0,0,0,None))
def get_grad_fn(weights, batch_stats, image, label, key, apply_fn):

    grad,aux = jax.grad(get_loss_fn,argnums=0,has_aux=True)(weights,batch_stats,image,label,key,apply_fn)

    return grad,aux

@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None))
def update_states_fn(weights, grad, opt_params, update_fn):
    updates,new_opt_params = update_fn(grad,opt_params,weights)
    new_weights = optax.apply_updates(weights,updates)

    return new_weights,new_opt_params

@partial(jax.jit,static_argnums=(6,7))
@partial(jax.vmap,in_axes=(0,0,0,0,0,0,None,None))
def train_step(weights, batch_stats, opt_params ,image, label, key, apply_fn, update_fn):

    grad,aux = jax.grad(get_loss_fn,argnums=0,has_aux=True)(weights,batch_stats,image,label,key,apply_fn)
    updates,new_opt_params = update_fn(grad,opt_params,weights)

    new_weights = optax.apply_updates(weights,updates)
    new_batch_stats = aux["batch_stats"]
    aux["grad"] = grad

    return new_weights,new_batch_stats,new_opt_params,aux



@partial(jax.vmap,in_axes=(0,None,None,None))
def conditional_tree_map(trees, cond_fn, true_fn, false_fn):
    return tree_map_with_path(lambda *e : true_fn(*e) if cond_fn(*e) else false_fn(*e),*trees)

def c_norm(x):
    return jnp.linalg.norm(x.reshape(-1,x.shape[-1]),axis=0)

def c_mean(x):
    return jnp.mean(x,axis=(0,1,2),keepdims=False)

def b_mean(x):
    return jnp.mean(x,axis=(0,1,3),keepdims=False)

def g_mean(x):
    return jnp.mean(x,axis=(0,1,2,3),keepdims=False)

def g_norm(x):
    return jnp.linalg.norm(x.reshape(-1),axis=0)

def c_norm_square(x):
    return c_norm(x)**2

@partial(jax.jit)
@partial(jax.vmap,in_axes=0)
def reverse_norms(weights):
    mean_norms = tree_map_with_path(lambda p,x : jnp.max(c_norm(x)) if "conv" in keystr(p).lower() and "kernel" in keystr(p).lower() else 0,weights)


    norm_mappings = [("Conv_0","Conv_7"),("Conv_1","Conv_6"),("Conv_2","Conv_5"),("Conv_3","Conv_4")]

    def get_new_weights(w,current_norm,new_norm):

        return new_norm*(w/(current_norm+1e-7))

    for conv1,conv2 in norm_mappings:
        weights[conv1]["kernel"] = get_new_weights(weights[conv1]["kernel"],mean_norms[conv1]["kernel"],mean_norms[conv2]["kernel"])
        weights[conv2]["kernel"] = get_new_weights(weights[conv2]["kernel"],mean_norms[conv2]["kernel"],mean_norms[conv1]["kernel"])

    return weights


@partial(jax.jit,static_argnums=(-1,))
def step_layer_weighted_division_norm_fn(w,c,n,N,L,mode="channel"):
    cond_fn = lambda s,_ : "conv" in keystr(s).lower() and "kernel" in keystr(s).lower()
    false_fn = lambda _,x : x

    if mode == "channel":
        def true_fn(s,x):
            layer_num=float(keystr(s).lower().split("conv_")[1].split("']")[0])+1
            step_layer_scale = ((N-n)/N)**((L-layer_num)/L)
            x =  (1-step_layer_scale)*x + step_layer_scale*x*(c/(c_norm(x)[None,None,None,:] + 1e-9))
            return x
    elif mode == "global":
        def true_fn(s,x):
            layer_num=float(keystr(s).lower().split("conv_")[1].split("']")[0])+1
            x =  x*((c/layer_num)/(g_norm(x) + 1e-9))

    return conditional_tree_map((w,),cond_fn,true_fn,false_fn)



@partial(jax.jit,static_argnums=(-1,))
def layer_weighted_division_norm_fn(w,c,n,N,L,mode="channel"):
    cond_fn = lambda s,_ : "conv" in keystr(s).lower() and "kernel" in keystr(s).lower()
    false_fn = lambda _,x : x
    if mode == "channel":
        def true_fn(s,x):
            layer_num=float(keystr(s).lower().split("conv_")[1].split("']")[0])+1
            x =  x*((c/layer_num)/(c_norm(x)[None,None,None,:] + 1e-9))
            return x
    elif mode == "global":
        def true_fn(s,x):
            layer_num=float(keystr(s).lower().split("conv_")[1].split("']")[0])+1
            x =  x*((c/layer_num)/(g_norm(x) + 1e-9))

    return conditional_tree_map((w,),cond_fn,true_fn,false_fn)

@partial(jax.jit,static_argnums=(-1,))
def division_norm_fn(w,c,n,N,L,mode="channel"):
    cond_fn = lambda s,_ : "conv" in keystr(s).lower() and "kernel" in keystr(s).lower()
    false_fn = lambda _,x : x
    if mode == "channel":
        true_fn = lambda _,x : x*(c/(c_norm(x)[None,None,None,:] + 1e-9))
    elif mode == "global":
        true_fn = lambda _,x : x*(c/(g_norm(x) + 1e-9))

    return conditional_tree_map((w,),cond_fn,true_fn,false_fn)

@partial(jax.jit,static_argnums=(-1,))
def center_division_norm_fn(w,c,n,N,L,mode="channel"):
    cond_fn = lambda s,_ : "conv" in keystr(s).lower() and "kernel" in keystr(s).lower()
    false_fn = lambda _,x : x
    if mode == "channel":
        true_fn = lambda _,x : c*(x-c_mean(x)[None,None,None,:])/(c_norm(x)[None,None,None,:] + 1e-7)
    elif mode == "global":
        true_fn = lambda _,x : c*(x-g_mean(x))/(g_norm(x) + 1e-7)

    return conditional_tree_map((w,),cond_fn,true_fn,false_fn)

@partial(jax.jit,static_argnums=(-1,))
def division_center_norm_fn(w,c,n,N,L,mode="channel"):
    cond_fn = lambda s,_ : "conv" in keystr(s).lower() and "kernel" in keystr(s).lower()
    false_fn = lambda _,x : x
    if mode == "channel":
        def true_fn(_,x):
            x = x/(c_norm(x)[None,None,None,:] + 1e-7)
            x = x - c_mean(x)[None,None,None,:]
            return c*x
    elif mode == "global":
        def true_fn(_,x):
            x = x/(g_norm(x) + 1e-7)
            x = x - g_mean(x)
            return c*x
        
    return conditional_tree_map((w,),cond_fn,true_fn,false_fn)

@partial(jax.vmap, in_axes=(0,None,None))
def init_model(key,model_input,init_fn):
    state = init_fn(key,model_input)
    weights,batch_stats = state["params"],state["batch_stats"]
    return weights,batch_stats

@partial(jax.vmap, in_axes=(0,None))
def init_optimizer(weights,init_fn):
    return init_fn(weights)

def get_states_device_put(model_init_fn,optim_init_fn,model_input,keys,default_cpu_device,named_sharding):
    # Initialize params on default_cpu_device
    with jax.default_device(default_cpu_device):
        weights,batch_stats = init_model(keys,model_input,model_init_fn)
        optimizer_state = init_optimizer(weights,optim_init_fn)

    # Distribute params on different devices
    weights,batch_stats = jax.device_put(weights,named_sharding),jax.device_put(batch_stats,named_sharding)
    optimizer_state = jax.device_put(optimizer_state,named_sharding)

    return weights,batch_stats,optimizer_state

def get_model(args):
    match args.model.model:
        case "vgg11":
            match args.model.activation_fn:
                case "relu":
                    activation_fn = nn.relu
                case "tanh":
                    activation_fn = nn.tanh
                case _:
                    exit("No matching activation_fn ({0}) found".format(args.model.activation_fn))
            model = VGG11(num_classes=args.model.num_classes,activation_fn = activation_fn)  
            num_conv_layers = 8
        case "resnet50":
            model = ResNet(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock,num_classes=args.model.num_classes)
            num_conv_layers = 38
        case _ :
            exit("No matching model ({0}) found".format(args.model.model))

    return model,num_conv_layers

def ds_stack_iterator(*ds):
    for ds_elems in zip(*ds):
        yield jnp.stack([e[0] for e in ds_elems]),jnp.stack([e[1] for e in ds_elems])    

def get_dataset(args):
    dataset_name = args.dataset.dataset
    data_dir = args.dataset.dataset_path
    batch_size = args.dataset.batch_size
    parallel_ds_needed = args.num_devices*args.num_experiments_per_device
    

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        ds_train = ds_stack_iterator(*[get_cifar(name=dataset_name,data_dir=data_dir,batch_size=batch_size)[0] for _ in range(parallel_ds_needed)])
        ds_test = ds_stack_iterator(*[get_cifar(name=dataset_name,data_dir=data_dir,batch_size=batch_size)[1] for _ in range(parallel_ds_needed)])

    else:
        exit("No matching dataset ({0}) found".format(args.dataset_name))

    return ds_train,ds_test

def get_optimizer(args,helper_weights):

    conv_kernel_wd_mask = tree_map_with_path(lambda p,_ : "conv" in keystr(p).lower() and "kernel" in keystr(p).lower(),helper_weights)

    lr = args.optimizer.lr
    lambda_wd = args.optimizer.lambda_wd

    wd = optax.add_decayed_weights(lambda_wd,mask=conv_kernel_wd_mask)
    match args.optimizer.optimizer:
        case "adam":
            optimizer = optax.chain(wd, optax.adam(learning_rate=lr)) 
        case "sgdm":
            optimizer = optax.chain(wd, optax.sgd(learning_rate=lr, momentum=args.optimizer.momentum))
        case _ :
            exit("No matching optimizer ({0}) found".format(args.optimizer.optimizer))

    return optimizer

def get_norm_fn(norm_fn):
    match norm_fn:
        case "channel_division":
            return lambda *x : division_norm_fn(*x,"channel")
        case "lw_channel_division":
            return lambda *x : layer_weighted_division_norm_fn(*x,"channel")
        case "slw_channel_division":
            return lambda *x : step_layer_weighted_division_norm_fn(*x,"channel") 
        case "channel_center_division":
            return lambda *x : center_division_norm_fn(*x,"channel")
        case "channel_division_center":
            return lambda *x : division_center_norm_fn(*x,"channel")
        case "channel_z":
            pass
        case "global_division":
            return lambda *x : division_norm_fn(*x,"global")
        case "global_center_division":
            return lambda *x : center_division_norm_fn(*x,"global")
        case "global_division_center":
            return lambda *x : division_center_norm_fn(*x,"global")
        case "global_z":
            pass
        case "identity":
            return lambda *x : x[0]


def dict_to_namespace(d):
    for key,value in d.items():
        if type(value) == dict:
            d[key] = dict_to_namespace(value)
    
    return SimpleNamespace(**d)


def namespace_to_dict(ns):
    d = vars(ns)
    for key,value in d.items():
        if type(value) == SimpleNamespace:
            d[key] = namespace_to_dict(value)
    
    return d
