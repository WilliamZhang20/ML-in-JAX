# ML_in_JAX

In this repository, I implement basic ML algorithms in JAX to become familiar with the library.
I also implemented several neural network convergence optimization techniques to learn more about them by coding them up from scratch!

Currently, the repository includes:
- a from-scratch implementation of a multi-layer perceptron (MLP) to recognize images handwritten digits. The data (from MNIST) is imported using PyTorch.
- a file called `train_utils.py` which contains vital functions for the neural network, such as the neural activation function, loss funciton, and forward propagation through layers.
- a file called `optimized_sgd.py` which contains optimziation algorithms for gradient descent, such as momentum, RMSProp, Adam, and Nesterov
- a file called `verify_vectors.py` which experiments with various configurations of a simple square operation on a vector. It shows that JAX runs much faster than typical numpy executions.

## Evaluating Gradient Descent Optimizations

Each training round took place over 10 epochs, each of which took no more than 5 seconds.

That meant that training was very fast, so it was easy to experiment with many parameters.

The original algorithm for gradient descent was run by in the `update` function in `train_utils.py`, following the classic formula $$\theta = \theta - \alpha \cdot \nabla J(\theta)$$, where $$\alpha$$ is learning rate and $$J$$ is the loss function.

After training 10 epochs with a learning rate of 0.01, the network's training dataset accuracy was 98.05%, and the test dataset accuracy was 97.20%.

The *momentum* optimization maintains a moving average of past gradients to smooth updates, which ultimately reduces the chance of getting stuck in local minima.

After training on momentum with a learning rate of 0.01, I obtained a training accuracy of 99.96%, and a test accuracy of 98.12%.

The RMSProp algorithm maintains an average of the second moment (variance) of gradients. However, I found it to be very hard to have the same training acceleration as momentum or any future algorithms...I need to do more work on it.

Adam (Adaptive Moments) combines both momentum and RMSProp. Its results were similarly disappointing, as it failed to reach high accuracy at the end of training.

Then I guessed something, which I realize was also true. The learning rate was too high. Adam adaptively scales the learning rate, by dividing it over a small number. There was no need to amplify it myself. 

Mathematically, the update of parameters (similar to RMSProp) goes like: 

$$
{\theta}_{t+1} = {\theta}_{t} - \frac{\alpha \cdot \hat{m_t}}{\sqrt{\hat{v_t}}  + \epsilon}
$$

So it was better to keep the learning rate around 0.001, which then yielded around 98% accuracy again!

Finally, the Nesterov Accelerated Gradient algorithm works in a similar fashion to momentum. 

Rather than simply calculate velocity from gradients and step in its direction, it first steps in the velocity's direction (the lookahead), and evaluates gradients using the lookahead. 

Then, it updates velocities from the resulting gradient and previous velocity, and updates parameters using the velocity.

So, mathematically:

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla J(\theta_t + \beta v_{t-1})
\theta_{t+1} = \theta_t - \alpha \cdot v_t
$$

After training *only* 10 epochs on a learning rate of 0.01, I achieved the best accuracy of 99.97% the on training dataset, and 98.26% on the cross-validating test dataset.