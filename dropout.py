import numpy as np
from jax import grad, jit
from jax import vmap
from jax import random
import jax
import jax.numpy as jnp

def dropout(x, rate, key=None, training=True):
    if training:
        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        x = jnp.multiply(x, mask) / keep_prob
    return x

def add_noise(x, noise_std, rng_key): 
    """
    Params:
    - x: input
    - noise_std: noise standard deviation
    - rng_key: random num gen key
    Returns: x with Gaussian Noise
    """
    noise = jax.random.normal(rng_key, shape=x.shape) * noise_std
    noisy_x = x + noise
    return noisy_x
