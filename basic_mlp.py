import numpy as np
from jax import grad, jit
from jax import vmap
from jax import random
import jax
import jax.numpy as jnp

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
batch_size = 128
n_targets = 10
params = init_network_params(layer_sizes, random.key(0))

from jax.scipy.special import logsumexp

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

random_flattened_image = random.normal(random.key(1), (28*28))
preds = predict(params, random_flattened_image)
print(preds.shape)

random_flattened_images = random.normal(random.key(1), (10, 28 * 28))
batched_predict = vmap(predict, in_axes=(None, 0))

batched_preds = batched_predict(params, random_flattened_images)
print(batched_preds.shape) # 10 logit values for each of 10 images so (10, 10)

def one_hot(x, k, dtype=jnp.float32):
    """create one-hot encoding of x of size k"""
    return jnp.array(x[:, None] == jnp.arange(k), dtype)

def accuracy(params, images, targets):
    target_class = jnp.argmax(targets, axis=1)
    predicted_class = jnp.argmax(batched_predict(params, images), axis=1)
    return jnp.mean(predicted_class == target_class)

def loss(params, images, targets):
    preds = batched_predict(params, images)
    return -jnp.mean(preds*targets)

@jit # accelerate gradient descent
def update(params, x, y):
    grads = grad(loss)(params, x, y)
    return [(w - step_size * dw, b-step_size * db)
            for (w, b), (dw, db) in zip(params, grads)]

import numpy as np
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

# training loop
import time

for epoch in range(num_epochs): # 10 epochs
    start_time = time.time()
    for x, y in training_generator:
        x = jnp.array(x)
        y = one_hot(jnp.array(y), n_targets)
        params = update(params, x, y) # accelerated with jit
    epoch_time = time.time() - start_time

    train_acc = accuracy(params, train_images, train_labels)
    test_acc = accuracy(params, test_images, test_labels)

    print("Epoch {} in {:0.2f} sec".format(epoch, epoch_time))
    print("Training set accuracy {}".format(train_acc))
    print("Test set accuracy {}".format(test_acc))
