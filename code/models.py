from flax import linen as nn
import jax 


from functools import partial
from collections.abc import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

class BatchNormIdentity(nn.Module):
    """
    Since flax requires the external handling of the batch_stats state, 
    and thus it has to be incorporated as a python variable, 
    this Identity function introduces a not used! batch_stats variable. 
    """
    @nn.compact
    def __call__(self, x):
        self.variable("batch_stats", "Empty", lambda s: jnp.zeros(s), (1,))
        return x 

class VGG11(nn.Module):
    num_classes: int
    activation_fn: any
    features_div : int
    use_bn : bool

    @nn.compact
    def __call__(self, x ,train: bool = True):
        x = nn.Conv(features=int(64/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=int(128/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=int(256/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=int(256/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=int(512/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=int(512/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=int(512/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=int(512/self.features_div),kernel_size=(3,3),padding=1,use_bias=not self.use_bn)(x)
        if self.use_bn:
            x = nn.BatchNorm(use_running_average=not train)(x)
        else:
            x = BatchNormIdentity()(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.avg_pool(x,window_shape=(1,1),strides=(1,1))

        x = x.reshape(x.shape[0],-1)

        x = nn.Dense(features=self.num_classes,name="out")(x)

        return x
    
    def get_layer_depth_dict(self):
        if self.use_bn:
            return {'BatchNorm_0': {'bias': 0, 'scale': 0},
            'BatchNorm_1': {'bias': 0, 'scale': 0},
            'BatchNorm_2': {'bias': 0, 'scale': 0},
            'BatchNorm_3': {'bias': 0, 'scale': 0},
            'BatchNorm_4': {'bias': 0, 'scale': 0},
            'BatchNorm_5': {'bias': 0, 'scale': 0},
            'BatchNorm_6': {'bias': 0, 'scale': 0},
            'BatchNorm_7': {'bias': 0, 'scale': 0},
            'Conv_0': {'kernel': 1},
            'Conv_1': {'kernel': 2},
            'Conv_2': {'kernel': 3},
            'Conv_3': {'kernel': 4},
            'Conv_4': {'kernel': 5},
            'Conv_5': {'kernel': 6},
            'Conv_6': {'kernel': 7},
            'Conv_7': {'kernel': 8},
            'out': {'bias': 0, 'kernel': 0}} 

        else:
            return {'Conv_0': {'kernel': 1, 'bias' : 1},
            'Conv_1': {'kernel': 2, 'bias' : 2},
            'Conv_2': {'kernel': 3, 'bias' : 3},
            'Conv_3': {'kernel': 4, 'bias' : 4},
            'Conv_4': {'kernel': 5, 'bias' : 5},
            'Conv_5': {'kernel': 6, 'bias' : 6},
            'Conv_6': {'kernel': 7, 'bias' : 7},
            'Conv_7': {'kernel': 8, 'bias' : 8},
            'out': {'bias': 0, 'kernel': 0}} 


