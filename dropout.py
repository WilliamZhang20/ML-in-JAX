import numpy as np
from jax import grad, jit
from jax import vmap
from jax import random
import jax
import jax.numpy as jnp

def dropout(x, rate, deterministic=False, key=None):
    if deterministic:
        return x
    
    mask = random.bernouilli(key, 1-rate, shape=x.shape)

    return x * mask / (1 - rate)

