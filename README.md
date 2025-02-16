# ML_in_JAX

In this repository, I implement basic ML algorithms in JAX to become familiar with the library.

Currently, the repository includes:
- a from-scratch implementation of a multi-layer perceptron (MLP) to recognize images handwritten digits. The data (from MNIST) is imported using PyTorch.
- a file called `train_utils.py` which contains vital functions for the neural network, such as the neural activation function, loss funciton, and forward propagation through layers.
- a file called `optimized_sgd.py` which contains optimziation algorithms for gradient descent, such as momentum, RMSProp, Adam, and Nesterov
- a file called "verify_vectors.py" which experiments with various configurations of a simple square operation on a vector. It shows that JAX runs much faster than typical numpy executions.