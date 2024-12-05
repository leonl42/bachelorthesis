from flax import linen as nn
import jax 

"""Flax implementation of ResNet V1.5."""

# See issue #620.
# pytype: disable=wrong-arg-count

from functools import partial
from typing import Any, Tuple
from collections.abc import Callable, Sequence

from flax import linen as nn
import jax.numpy as jnp

class BottleneckResNetBlock(nn.Module):
  """Bottleneck ResNet block."""

  filters: int
  conv: Any
  norm: Any
  act: Callable
  strides: tuple[int, int] = (1, 1)

  @nn.compact
  def __call__(self, x, i):
    residual = x
    y = self.conv(self.filters, (1, 1), name = "conv_" + str(i))(x)
    y = self.norm(name = "norm_" + str(i))(y)
    y = self.act(y)

    i += 1
    y = self.conv(self.filters, (3, 3), self.strides, name = "conv_" + str(i))(y)
    y = self.norm(name = "norm_" + str(i))(y)
    y = self.act(y)

    i += 1
    y = self.conv(self.filters * 4, (1, 1), name = "conv_" + str(i))(y)
    y = self.norm(scale_init=nn.initializers.zeros_init(),name = "norm_" + str(i))(y)

    if residual.shape != y.shape:
      i += 1
      residual = self.conv(
          self.filters * 4, (1, 1), self.strides, name="conv_" + str(i)
      )(residual)
      residual = self.norm(name="norm_" + str(i))(residual)

    return (self.act(residual + y),i)


class ResNet(nn.Module):
  """ResNetV1.5."""

  stage_sizes: Sequence[int]
  block_cls: Any
  num_classes: int
  num_filters: int = 64
  dtype: Any = jnp.float32
  act: Callable = nn.relu
  conv: Any = nn.Conv

  @nn.compact
  def __call__(self, x, train: bool = True):
    conv = partial(self.conv, use_bias=False, dtype=self.dtype)
    norm = partial(
        nn.BatchNorm,
        use_running_average=not train,
        momentum=0.9,
        epsilon=1e-5,
        dtype=self.dtype,
        axis_name='batch',
    )

    x = conv(
        self.num_filters,
        (7, 7),
        (2, 2),
        padding=[(3, 3), (3, 3)],
        name='conv_0',
    )(x)
    x = norm(name='bn_0')(x)
    x = nn.relu(x)
    x = nn.max_pool(x, (3, 3), strides=(2, 2), padding='SAME')

    name_index = 1
    for i, block_size in enumerate(self.stage_sizes):
      for j in range(block_size):
        strides = (2, 2) if i > 0 and j == 0 else (1, 1)
        x,name_index = self.block_cls(
            self.num_filters * 2**i,
            strides=strides,
            conv=conv,
            norm=norm,
            act=self.act,
        )(x,name_index)
    x = jnp.mean(x, axis=(1, 2))
    x = nn.Dense(self.num_classes, dtype=self.dtype)(x)
    x = jnp.asarray(x, self.dtype)
    return x


class VGG11(nn.Module):
    num_classes: int
    activation_fn: any

    @nn.compact
    def __call__(self, x ,train: bool = True):
        x = nn.Conv(features=64,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=128,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=256,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=256,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)

        x = nn.Conv(features=512,kernel_size=(3,3),padding=1,use_bias=False)(x)
        x = nn.BatchNorm(use_running_average=not train)(x)
        x = self.activation_fn(x)
        x = nn.max_pool(x,window_shape=(2,2),strides=(2,2))

        x = nn.avg_pool(x,window_shape=(1,1),strides=(1,1))

        x = x.reshape(x.shape[0],-1)

        x = nn.Dense(features=self.num_classes,name="out")(x)

        return x


class Extendable(nn.Module):
    num_classes: int
    num_layers: int

    @nn.compact
    def __call__(self, x ,train: bool = True):
        for _ in range(self.num_layers):
            x = nn.Conv(features=64,kernel_size=(3,3),use_bias=False,padding="SAME")(x)
            x = nn.BatchNorm(use_running_average=not train)(x)
            x = nn.relu(x)
        
        x = nn.avg_pool(x,window_shape=(1,1),strides=(1,1))
        x = x.reshape(x.shape[0],-1)
        x = nn.Dense(features=self.num_classes,name="out")(x)

        return x
    

layer_depth_resnet50 = {'BottleneckResNetBlock_0': {'conv_1': {'kernel': 2},
'conv_2': {'kernel': 3},
'conv_3': {'kernel': 4},
'conv_4': {'kernel': 5},
'norm_1': {'bias': 0, 'scale': 0},
'norm_2': {'bias': 0, 'scale': 0},
'norm_3': {'bias': 0, 'scale': 0},
'norm_4': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_1': {'conv_4': {'kernel': 5},
'conv_5': {'kernel': 6},
'conv_6': {'kernel': 7},
'norm_4': {'bias': 0, 'scale': 0},
'norm_5': {'bias': 0, 'scale': 0},
'norm_6': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_10': {'conv_24': {'kernel': 25},
'conv_25': {'kernel': 26},
'conv_26': {'kernel': 27},
'norm_24': {'bias': 0, 'scale': 0},
'norm_25': {'bias': 0, 'scale': 0},
'norm_26': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_11': {'conv_26': {'kernel': 27},
'conv_27': {'kernel': 28},
'conv_28': {'kernel': 29},
'norm_26': {'bias': 0, 'scale': 0},
'norm_27': {'bias': 0, 'scale': 0},
'norm_28': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_12': {'conv_28': {'kernel': 29},
'conv_29': {'kernel': 30},
'conv_30': {'kernel': 31},
'norm_28': {'bias': 0, 'scale': 0},
'norm_29': {'bias': 0, 'scale': 0},
'norm_30': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_13': {'conv_30': {'kernel': 31},
'conv_31': {'kernel': 32},
'conv_32': {'kernel': 33},
'conv_33': {'kernel': 34},
'norm_30': {'bias': 0, 'scale': 0},
'norm_31': {'bias': 0, 'scale': 0},
'norm_32': {'bias': 0, 'scale': 0},
'norm_33': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_14': {'conv_33': {'kernel': 34},
'conv_34': {'kernel': 35},
'conv_35': {'kernel': 36},
'norm_33': {'bias': 0, 'scale': 0},
'norm_34': {'bias': 0, 'scale': 0},
'norm_35': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_15': {'conv_35': {'kernel': 36},
'conv_36': {'kernel': 37},
'conv_37': {'kernel': 38},
'norm_35': {'bias': 0, 'scale': 0},
'norm_36': {'bias': 0, 'scale': 0},
'norm_37': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_2': {'conv_6': {'kernel': 7},
'conv_7': {'kernel': 8},
'conv_8': {'kernel': 9},
'norm_6': {'bias': 0, 'scale': 0},
'norm_7': {'bias': 0, 'scale': 0},
'norm_8': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_3': {'conv_10': {'kernel': 11},
'conv_11': {'kernel': 12},
'conv_8': {'kernel': 9},
'conv_9': {'kernel': 10},
'norm_10': {'bias': 0, 'scale': 0},
'norm_11': {'bias': 0, 'scale': 0},
'norm_8': {'bias': 0, 'scale': 0},
'norm_9': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_4': {'conv_11': {'kernel': 12},
'conv_12': {'kernel': 13},
'conv_13': {'kernel': 14},
'norm_11': {'bias': 0, 'scale': 0},
'norm_12': {'bias': 0, 'scale': 0},
'norm_13': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_5': {'conv_13': {'kernel': 14},
'conv_14': {'kernel': 15},
'conv_15': {'kernel': 16},
'norm_13': {'bias': 0, 'scale': 0},
'norm_14': {'bias': 0, 'scale': 0},
'norm_15': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_6': {'conv_15': {'kernel': 16},
'conv_16': {'kernel': 17},
'conv_17': {'kernel': 18},
'norm_15': {'bias': 0, 'scale': 0},
'norm_16': {'bias': 0, 'scale': 0},
'norm_17': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_7': {'conv_17': {'kernel': 18},
'conv_18': {'kernel': 19},
'conv_19': {'kernel': 20},
'conv_20': {'kernel': 21},
'norm_17': {'bias': 0, 'scale': 0},
'norm_18': {'bias': 0, 'scale': 0},
'norm_19': {'bias': 0, 'scale': 0},
'norm_20': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_8': {'conv_20': {'kernel': 21},
'conv_21': {'kernel': 22},
'conv_22': {'kernel': 23},
'norm_20': {'bias': 0, 'scale': 0},
'norm_21': {'bias': 0, 'scale': 0},
'norm_22': {'bias': 0, 'scale': 0}},
'BottleneckResNetBlock_9': {'conv_22': {'kernel': 23},
'conv_23': {'kernel': 24},
'conv_24': {'kernel': 25},
'norm_22': {'bias': 0, 'scale': 0},
'norm_23': {'bias': 0, 'scale': 0},
'norm_24': {'bias': 0, 'scale': 0}},
'Dense_0': {'bias': 0, 'kernel': 0},
'bn_0': {'bias': 0, 'scale': 0},
'conv_0': {'kernel': 1}}

layer_depth_vgg11 = {'BatchNorm_0': {'bias': 0, 'scale': 0},
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