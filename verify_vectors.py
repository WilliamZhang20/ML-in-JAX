import time
import jax.numpy as jnp
from jax import jit
import numpy as np

# standard function
def square(x):
    return x ** 2

# build huge array of numbers
x = jnp.arange(1e7)

# vectorized time - 
start = time.time()
y_vectorized = square(x)
end = time.time()
print("JAX vectorized time:", end-start)

# JAX vectorized time (on jit)
square_jit = jit(square)
start = time.time()
y_vectorized = square_jit(x).block_until_ready()  # Synchronize to get accurate timing
end = time.time()
print("JAX vectorized time (with JIT):", end - start)

# fastest approach - lazy on jax array
start = time.time()
y_lazy = x ** 2
end = time.time()
print("JAX lazy time:", end-start)

# numpy-only based approach - the slowest
x = np.arange(1e7)
start = time.time()
y_numpy = x**2
end = time.time()
print("numpy based time:", end-start)