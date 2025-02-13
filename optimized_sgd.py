import numpy as np
from jax import grad, jit
import jax
import jax.numpy as jnp
from train_utils import loss

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
    new_velocities = [(momentum * v + step_size * g) for v, g in zip(velocities, grads)]
    new_params = [(w - v_w, b - v_b) for (w, b), (v_w, v_b) in zip(params, new_velocities)]
    return new_params, new_velocities

@jit
def rmsprop(params, x, y, squared_grads, step_size=0.01, decay=0.99, epsilon=1e-8):
    grads = grad(loss)(params, x, y)

    new_squared_grads = [(decay * sg + (1 - decay) * (g ** 2)) for sg, g in zip(squared_grads, grads)]
    new_params = [(w - (step_size / (jnp.sqrt(sg_w + epsilon))) * g_w,
                   b - (step_size / (jnp.sqrt(sg_b + epsilon))) * g_b)
                  for (w, b), (sg_w, sg_b), (g_w, g_b) in zip(params, new_squared_grads, grads)]
    
    return new_params, new_squared_grads