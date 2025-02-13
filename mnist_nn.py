import numpy as np

import jax
from jax import grad, jit
from jax import vmap
from jax import random
import jax.numpy as jnp
from train_utils import batched_predict, update

# random weights & biases
def random_layer_params(m, n, key, scale=1e-2):
    w_key, b_key = random.split(key)
    return scale * random.normal(w_key, (n, m)), scale * random.normal(b_key, (n, ))

# initialize layers
def init_network_params(sizes, key):
    keys = random.split(key, len(sizes))
    return [random_layer_params(m, n, k) for m, n, k in zip(sizes[:-1], sizes[1:], keys)]

layer_sizes = [784, 512, 512, 10]
step_size = 0.01
num_epochs = 10
batch_size = 128 # for mini-batched GD
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))


def one_hot(x, k, dtype=jnp.float32):
    """create one-hot encoding of x of size k"""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

from jax.tree_util import tree_map
from torch.utils import data
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

def numpy_collate(batch):
    return tree_map(np.asarray, data)

transform = transforms.Compose([transforms.Lambda(lambda x: np.ravel(np.array(x, dtype=np.float32)))])

# Load MNIST dataset with transformation
mnist_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
mnist_dataset_test = MNIST(root='./data', train=False, download=True, transform=transform)

train_images = np.array(mnist_dataset.data.numpy().reshape(len(mnist_dataset.data), -1), dtype=np.float32)
train_labels = one_hot(jnp.array(mnist_dataset.targets), n_targets)

test_images = np.array(mnist_dataset_test.data.numpy().reshape(len(mnist_dataset_test.data), -1), dtype=np.float32)
test_labels = one_hot(jnp.array(mnist_dataset_test.targets), n_targets)

training_generator = data.DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)

from optimized_sgd import sgd_momentum

# Initialize velocities - a matrix
velocities = [(jnp.zeros_like(w), jnp.zeros_like(b)) for w, b in params]

# training loop
import time

for epoch in range(num_epochs): # 10 epochs
    start_time = time.time()
    for x, y in training_generator:
        x = jnp.array(x)
        y = one_hot(jnp.array(y), n_targets)
        params, velocities = sgd_momentum(params, x, y, velocities, step_size, momentum=0.9)
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
