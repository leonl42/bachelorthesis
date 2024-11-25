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