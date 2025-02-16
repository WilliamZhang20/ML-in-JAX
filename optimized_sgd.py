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
def rmsprop(params, x, y, squared_grads, step_size=0.01, decay=0.88, epsilon=1e-8):
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

@jit
def adam(params, x, y, m, v, t, step_size=0.001, beta1=0.9, beta2=0.98, epsilon=1e-10):
    grads = grad(loss)(params, x, y)
    # t: timestep (to help with bias correction)
    t += 1
    # Update biased first moment estimate
    new_m = [
        (beta1 * m_w + (1 - beta1) * g_w, beta1 * m_b + (1 - beta1) * g_b)
        for (m_w, m_b), (g_w, g_b) in zip(m, grads)
    ]
    # Update biased second moment estimate
    new_v = [
        (beta2 * v_w + (1 - beta2) * (g_w ** 2), beta2 * v_b + (1 - beta2) * (g_b ** 2))
        for (v_w, v_b), (g_w, g_b) in zip(v, grads)
    ]
    # Bias correction
    m_hat = [
        (m_w / (1 - beta1**t), m_b / (1 - beta1**t))
        for m_w, m_b in new_m
    ]
    v_hat = [
        (v_w / (1 - beta2**t), v_b / (1 - beta2**t))
        for v_w, v_b in new_v
    ]
    # Update parameters using corrected moments
    new_params = [
        (w - step_size * m_hat_w / (jnp.sqrt(v_hat_w) + epsilon),
         b - step_size * m_hat_b / (jnp.sqrt(v_hat_b) + epsilon))
        for (w, b), (m_hat_w, m_hat_b), (v_hat_w, v_hat_b) in zip(params, m_hat, v_hat)
    ]
    return new_params, new_m, new_v, t

@jit
def nesterov(params, x, y, v, step_size=0.001, momentum=0.9):
    lookahead = [ # lookahead guess given velocities!
        (w - momentum * v_w, b - momentum * v_b)
        for (w, b), (v_w, v_b) in zip(params, v)
    ]
    grads = grad(loss)(lookahead, x, y) # gradient with respect to lookahead prediction - so accelerated convergence!
    new_v = [
        (momentum * v_w + step_size * g_w, momentum * v_b + step_size * g_b)
        for (v_w, v_b), (g_w, g_b) in zip(v, grads)
    ]
    new_params = [
        (w - v_w, b - v_b) for (w, b), (v_w, v_b) in zip(params, new_v)
    ]
    return new_params, new_v