# ML_in_JAX

In this repository, I implement basic ML algorithms in JAX to become familiar with the library.

Currently, the repository includes:
- implementing a multi-layer perceptron (MLP) to guess images handwritten digits. The data (from MNIST) is imported using PyTorch. 
- a file called "verify_vectors.py" which experiments with various configurations of a simple square operation on a vector. It shows that JAX runs much faster than typical numpy executions.

Next up:
- implementing dropout on a basic MLP with noisy data
- building convolutional neural networks
- implementing an RL algorithm (e.g. REINFORCE policy gradient)