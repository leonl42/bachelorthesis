import jax
import jax.numpy as jnp
import optax
from functools import partial
from models import VGG11,VGG11_slim,ResNet,BottleneckResNetBlock
import tensorflow as tf
import tensorflow_datasets as tfds
from jax.tree_util import tree_map_with_path,keystr,tree_map,tree_leaves_with_path,tree_leaves
import flax.linen as nn
import copy
from types import SimpleNamespace

@partial(jax.jit,static_argnums=(4,))
@partial(jax.vmap,in_axes=(0,0,0,0,None))
def test_step(weights, batch_stats, image, label, apply_fn):
    """
    Performs a test step and returns the stats (loss/acc).

    Args:
        - weights <dict> : Weights of the network
        - batch_stats <dict> : Current batch statistics of the network
        - image <jax.Array> : input to the network
        - label <jax.Array> : target for the loss/acc calculation
        - apply_fn <function> : Function that takes the input, weights and batch stats and performs a forward step
    Returns:
        - aux <dict> : loss and accuracy
    """

    prediction = apply_fn({"params" : weights, "batch_stats" : batch_stats} ,image, train=False)
    
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(prediction,label))
    acc = jnp.mean((jnp.argmax(prediction,axis=-1) == label)*1)

    return {"loss" : loss, "acc" : acc}

def eval(weights,batch_stats,apply_fn,ds,its):
    """
    Calculates stats (loss/acc) on a dataset and returns the average
    Args:
        - weights <dict> : Weights of the network
        - batch_stats <dict> : Current batch statistics of the network
        - apply_fn <function> : Function that takes the input, weights and batch stats and performs a forward step
        - ds <Iterator> : Iterator containing (image,label) pairs
        - its <Int> : Over how many iterations to calculate the stats
    Returns:
        - aux <dict> : loss and accuracy averaged over the number of steps taken on the dataset
    """

    aux_agg = []

    for _,(img,lbl) in zip(range(its),ds):
        aux = test_step(weights,batch_stats,img,lbl,apply_fn)
        aux_agg.append(aux)

    return tree_map(lambda *x : jnp.mean(jnp.stack(x,axis=0),axis=0),*aux_agg)


def get_loss_fn(weights, batch_stats, image, label, key, apply_fn):
    """
    Performs a train step and returns the loss and the stats (loss/acc/batch statistics).

    Args:
        - weights <dict> : Weights of the network
        - batch_stats <dict> : Current batch statistics of the network
        - image <jax.Array> : input to the network
        - label <jax.Array> : target for the loss/acc calculation
        - key <jax.Array> : Random key
        - apply_fn <function> : Function that takes the input, weights and batch stats and performs a forward step
    Returns:
        - loss <jax.Array> : Loss
        - aux <dict> : contains loss, accuracy and updated batch statistics
    """

    prediction,updates = apply_fn({"params" : weights, "batch_stats" : batch_stats}, image, train=True, mutable=["batch_stats"],rngs={"dropout" : key})
    
    loss = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(prediction,label))
    acc = jnp.mean((jnp.argmax(prediction,axis=-1) == label)*1)

    return loss,{"loss" : loss, "acc" : acc, "batch_stats" : updates["batch_stats"]}


@partial(jax.jit,static_argnums=5)
@partial(jax.vmap,in_axes=(0,0,0,0,0,None))
def get_grad_fn(weights, batch_stats, image, label, key, apply_fn):
    """
    Performs a train step and returns the gradient and the stats (loss/acc/batch statistics).

    Args:
        - weights <dict> : Weights of the network
        - batch_stats <dict> : Current batch statistics of the network
        - image <jax.Array> : input to the network
        - label <jax.Array> : target for the loss/acc calculation
        - key <jax.Array> : Random key
        - apply_fn <function> : Function that takes the input, weights and batch stats and performs a forward step
    Returns:
        - grad <dict> : Gradient of the weights
        - aux <dict> : contains loss, accuracy and updated batch statistics
    """

    grad,aux = jax.grad(get_loss_fn,argnums=0,has_aux=True)(weights,batch_stats,image,label,key,apply_fn)

    return grad,aux

@partial(jax.jit,static_argnums=3)
@partial(jax.vmap,in_axes=(0,0,0,None))
def update_states_fn(weights, grad, opt_params, update_fn):
    """
    Perform optimizer step with the weights and grads and return new weights and new optimizer state.
    Args:
        - weights <dict> : Weights of the network
        - grad <dict> : Gradient of the weights of the network
        - opt_params <dict> : Parameters of the optimizer
        - update_fn <Function> : Function that takes in weights,gradient and optimizer params and 
                                returns the updates and new optimizer params. The updates are applied 
                                to the weights to get the new weights.
    Returns:
        - new_weights <dict> : Updated weight of the network
        - new_opt_params <dict> : Updated optimizer parameters
    """
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

def r_mean(x):
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
@partial(jax.vmap,in_axes=(0,None,0))
def weight_global_center_std_uncenter(w,scale,target_std):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense layer of shape [inC,outC]
        scale <float> : scale parameter
        std <float> : target standard deviation
    Returns: 
        w <jax.Array> : Weight matrix but with the weight means normalized to 0 and the weight std normalized to "scale*target_std".
    """
    # Compute the channel means
    mean = jnp.mean(w,keepdims=True)

    # Compute the weight matrix with channel means normalized to 0
    w = (w-mean)

    # Compute the channel stds
    std = jnp.std(w,keepdims=True)

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
def weight_reverse_center_normalize_uncenter(w,scale):
    """
    Takes:
        w <jax.array> : Weight matrix of a dense or conv layer
        scale <float> : scale parameter
    Returns: 
        w <jax.Array> : Weight matrix but with the input means normalized to 0, the channel norms normalized to "scale" and channel means scaled back to original mean.
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
    w = scale*w/(norm+1e-7) + mean

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

@partial(jax.vmap, in_axes=(0,None,None))
def init_model(key,model_input,init_fn):
    state = init_fn(key,model_input)
    weights,batch_stats = state["params"],state["batch_stats"]
    return weights,batch_stats

@partial(jax.vmap, in_axes=(0,None))
def init_optimizer(weights,init_fn):
    return init_fn(weights)

def get_states(model_init_fn,optim_init_fn,model_input,keys,default_cpu_device):
    # Initialize params on default_cpu_device
    with jax.default_device(default_cpu_device):
        weights,batch_stats = init_model(keys,model_input,model_init_fn)
        optimizer_state = init_optimizer(weights,optim_init_fn)

    return weights,batch_stats,optimizer_state

def device_put(named_sharding,*x):
    return [jax.device_put(e,named_sharding) for e in x]

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
            if args.model.features_div:
                features_div = args.model.features_div
            else:
                features_div = 1
            
            if not args.model.use_bn is None:
                use_bn = args.model.use_bn
            else:
                use_bn = True

            model = VGG11(num_classes=args.model.num_classes,activation_fn = activation_fn, features_div = features_div,use_bn=use_bn) 
            l = model.get_layer_depth_dict()
            L = 8
        case "resnet50":
            model = ResNet(stage_sizes=[3, 4, 6, 3], block_cls=BottleneckResNetBlock,num_classes=args.model.num_classes)
            l = None
            L = 38
        case _ :
            exit("No matching model ({0}) found".format(args.model.model))

    return model,l,L

tfds.display_progress_bar(enable=False)

def get_cifar(name,data_dir="./datasets/",batch_size = 128, shuffle_buffer = 5000, prefetch= tf.data.AUTOTUNE,repeats=-1, bfloat16 = False, cutoff_train_set = None,cutoff_test_set = None):
    builder = tfds.builder(name,data_dir=data_dir)
    builder.download_and_prepare()
    ds_train,ds_test = builder.as_dataset(split=["train", "test"])

    solve_dict = lambda elem : (elem["image"],elem["label"])
    ds_train,ds_test = ds_train.map(solve_dict),ds_test.map(solve_dict)

    dtype = tf.dtypes.bfloat16 if bfloat16 else tf.dtypes.float32
    cast = lambda img,lbl : (tf.cast(img,dtype),lbl)
    ds_train,ds_test = ds_train.map(cast),ds_test.map(cast)

    mean = tf.convert_to_tensor([0.32768, 0.32768, 0.32768],dtype=dtype)[None,None,:]
    std = tf.convert_to_tensor([0.27755222, 0.26925606, 0.2683012 ],dtype=dtype)[None,None,:]

    normalize = lambda img,lbl : ((img/255-mean)/std,lbl)

    ds_train,ds_test = ds_train.map(normalize),ds_test.map(normalize)

    if cutoff_train_set:
        ds_train = ds_train.shuffle(buffer_size=50000).take(cutoff_train_set).cache()
    if cutoff_test_set:
        ds_test = ds_test.shuffle(buffer_size=10000).take(cutoff_test_set).cache()

    prepare = lambda ds : ds.repeat(repeats).shuffle(buffer_size=shuffle_buffer).batch(batch_size,drop_remainder=True).prefetch(prefetch)
    ds_train,ds_test = prepare(ds_train), prepare(ds_test)

    return ds_train.as_numpy_iterator(),ds_test.as_numpy_iterator()


def ds_stack_iterator(*ds):
    """
    Stack multiple independent datasets for parallel training
    """
    for ds_elems in zip(*ds):
        yield jnp.stack([e[0] for e in ds_elems]),jnp.stack([e[1] for e in ds_elems])    

def get_dataset(args,bfloat16=False):
    dataset_name = args.dataset.dataset
    data_dir = args.dataset.dataset_path
    batch_size = args.dataset.batch_size
    parallel_ds_needed = args.num_devices*args.num_experiments_per_device
    cutoff_train_set = args.dataset.cutoff_train_set
    cutoff_test_set = args.dataset.cutoff_test_set

    if dataset_name == "cifar10" or dataset_name == "cifar100":
        ds_train = ds_stack_iterator(*[get_cifar(name=dataset_name,data_dir=data_dir,batch_size=batch_size,bfloat16=bfloat16,cutoff_train_set=cutoff_train_set)[0] for _ in range(parallel_ds_needed)])
        ds_test = ds_stack_iterator(*[get_cifar(name=dataset_name,data_dir=data_dir,batch_size=batch_size,bfloat16=bfloat16,cutoff_test_set=cutoff_test_set)[1] for _ in range(parallel_ds_needed)])
    else:
        exit("No matching dataset ({0}) found".format(args.dataset_name))

    return ds_train,ds_test

def get_optimizer(args,helper_weights):

    if not args.optimizer.apply_wd_to:
        apply_wd_to = "conv&kernel"
    else:
        apply_wd_to = args.optimizer.apply_wd_to

    conv_kernel_wd_mask = tree_map_with_path(lambda s,_ : substrings_in_path(s,apply_wd_to),helper_weights)

    if not args.optimizer.lr_scheduler:
        lr = args.optimizer.lr
    elif args.optimizer.lr_scheduler.type == "random":
        lr = lambda step : jax.random.uniform(key=jax.random.key(step),minval=args.optimizer.lr_scheduler.minval,maxval=args.optimizer.lr_scheduler.maxval)

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
        case "norm":
            return weight_normalize
        case "center_norm":
            return weight_center_normalize
        case "center_norm_uncenter":
            return weight_center_normalize_uncenter 
        case "center_std_uncenter":
            return weight_center_std_uncenter
        case "global_center_std_uncenter":
            return weight_global_center_std_uncenter
        case "reverse_center_normalize":
            return weight_reverse_center_normalize
        case "reverse_center_normalize_uncenter":
            return weight_reverse_center_normalize_uncenter
        case "identity":
            return lambda *x : x[0]
        
def get_change_scale(change_scale):
    match change_scale:
        case "layerwise_stepwise":
            return lambda n,N,l,L : ((N-n)/N)**((L-l)/L)
        case "layerwise":
            return lambda n,N,l,L : ((L-l)/L)
        case "stepwise":
            return lambda n,N,l,L : ((N-n)/N)
        case "identity":
            return lambda n,N,l,L : 1

class SimpleNamespaceNone(SimpleNamespace):
    # NameSpace that returns None instead of throwing an error when an undefined name is accessed

    def __getattr__(self, _):
        return None

def dict_to_namespace(d):
    #Transform a dictionary into a namespace

    for key,value in d.items():
        if type(value) == dict:
            d[key] = dict_to_namespace(value)
    
    return SimpleNamespaceNone(**d)


def namespace_to_dict(ns):
    #Transform a namespace into a dictionary

    out_dict = {}
    d = vars(ns)
    for key,value in d.items():
        if isinstance(value,SimpleNamespace) or isinstance(value,SimpleNamespaceNone):
            out_dict[key] = namespace_to_dict(value)
        else:
            out_dict[key] = value
    
    return out_dict


def substrings_in_path(s,match):
    """
    Check if certain keywords appear in a path by jax.tree_utils.tree_map_with_path
    
    Args:
        - s <list>: List of Strings that represent a dictionary path
        - match <str>: String of the following format: "a&b&...|c&d&... | ...", where a,b,c,d,... are Strings.

    Returns:
        - <bool>: True if [(a and b and ...) or (c and d and ...) or (...)] are inside s.
    """

    ors = match.split("|")
    ands = [e_or.split("&") for e_or in ors]

    return any([all([e_and.lower() in keystr(s).lower() for e_and in e_or]) for e_or in ands])
