import numpy as np
from jax import grad, jit
import jax
import jax.numpy as jnp
from train_utils import loss # for now, is cross entropy loss!

"""
Applying optimizations to gradient descent such as:
- Momentum with SGD
- RMSProp
- Learning Rate Scheduling
- Adam
- Nesterov
"""

@jit
def sgd_momentum(params, x, y, velocities, step_size, momentum=0.9):
    grads = grad(loss)(params, x, y)
    new_velocities = [
        (momentum * v_w + step_size * g_w, momentum * v_b + step_size * g_b)
        for (v_w, v_b), (g_w, g_b) in zip(velocities, grads)
    ]
    new_params = [
        (w - v_w, b - v_b) for (w, b), (v_w, v_b) in zip(params, new_velocities)
    ]
    return new_params, new_velocities

@jit
def rmsprop(params, x, y, squared_grads, step_size=0.01, decay=0.9, epsilon=1e-8):
    grads = grad(loss)(params, x, y)

    new_squared_grads = [ # also considered to be the second moment from statistics
        (decay * c_w + (1 - decay) * (g_w ** 2),
         decay * c_b + (1 - decay) * (g_b ** 2))
        for (c_w, c_b), (g_w, g_b) in zip(squared_grads, grads)
    ]
    new_params = [
        (w - (step_size / (jnp.sqrt(c_w) + epsilon)) * g_w,
         b - (step_size / (jnp.sqrt(c_b) + epsilon)) * g_b)
        for (w, b), (c_w, c_b), (g_w, g_b) in zip(params, new_squared_grads, grads)
    ]

    return new_params, new_squared_grads
