import numpy as np
from jax import grad, jit
from jax import vmap
from jax import random
import jax
import jax.numpy as jnp

from jax.scipy.special import logsumexp

@jit # update conducting gradient descent
def update(params, x, y, step_size):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b-step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

def relu(x):
    return jnp.maximum(0, x)

def predict(params, image):
    activations = image
    for w, b in params[:-1]:
        outputs = jnp.dot(w, activations) + b
        activations = relu(outputs)

    final_w, final_b = params[-1]
    logits = jnp.dot(final_w, activations) + final_b
    return logits - logsumexp(logits)

# Initialize a vectorized function to predict batches
batched_predict = vmap(predict, in_axes=(None, 0))

def dropout(x, rate, key=None, training=True):
    if training:
        keep_prob = 1.0 - rate
        mask = jax.random.bernoulli(key, p=keep_prob, shape=x.shape)
        x = jnp.multiply(x, mask) / keep_prob
    return x

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds*targets) # jnp.mean() is final step of cross-entropy loss


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
