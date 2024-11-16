# Neural Network-Optimizers

## Overview
This project demonstrates the application of various optimization algorithms to train a Convolutional Neural Network (CNN) on the MNIST dataset. While classification is the end task, the primary focus of this repository is to explore and compare the behavior of different optimizers during training.

## Aim
The main aim of this project is to provide an intuitive understanding of the following optimization algorithms by observing their training dynamics and performance trends:

1. **Adam**: Combines the benefits of RMSprop and momentum, maintaining adaptive learning rates for each parameter and accelerating convergence in sparse gradients.
2. **SGD (Stochastic Gradient Descent)**: A simple optimizer that updates weights using gradients calculated from individual batches. It serves as a baseline for comparison.
3. **SGD with Momentum**: Improves SGD by adding momentum, which helps accelerate convergence in the relevant direction while reducing oscillations.
4. **Adagrad**: Adapts learning rates individually for each parameter based on the sum of historical squared gradients, making it suitable for sparse data.
5. **RMSprop**: Modifies Adagrad by introducing a decay factor to the historical gradients, addressing Adagrad's diminishing learning rates over time.

By visualizing loss trends for each optimizer, this repository helps users understand their differences in speed, stability, and effectiveness during training.

## Features
- **Optimizer Comparison**: Observe how different optimizers behave in terms of convergence and stability.
- **Loss Visualization**: Generate and save loss plots for each optimizer to visually compare training performance.
- **Simple CNN Architecture**: A straightforward model to highlight the role of optimizers rather than the complexity of the network.
- 
## Requirements
Ensure you have the following installed:
- Python 3.7+
- PyTorch
- torchvision
- matplotlib

Install the dependencies using:
```bash
pip install torch torchvision matplotlib
```

## Dataset
The MNIST dataset, a standard benchmark for image classification, is used here. It contains 60,000 training and 10,000 test grayscale images of handwritten digits (0-9). The dataset is automatically downloaded through PyTorch's `torchvision.datasets.MNIST` module.

## Code Structure
1. **Dataset Preparation**  
   - The dataset is loaded with the `ToTensor()` transformation to normalize pixel values.

2. **CNN Model**  
   - A basic two-layer CNN is used for experimentation.

3. **Optimizers**  
   - The project demonstrates the performance of:
     - Adam
     - SGD
     - SGD with Momentum
     - Adagrad
     - RMSprop

4. **Training Function**  
   - Trains the model for a specified number of epochs, logging the loss and updating weights.

5. **Testing Function**  
   - Evaluates the test accuracy after training.

6. **Loss Visualization**  
   - Loss plots for each optimizer are saved for analysis.

## Expected Output
1. **Test Accuracy**  
   The accuracy of the CNN on the MNIST test set for each optimizer.

2. **Loss Plots**  
   Visualizations showing the loss reduction over epochs for each optimizer.

## Optimizer Descriptions
### Adam (Adaptive Moment Estimation)
- Combines momentum and RMSprop.
- Uses adaptive learning rates for each parameter.
- Well-suited for sparse gradients and non-stationary problems.
  
![AdamOptimizer](https://github.com/user-attachments/assets/e6563913-7433-496e-a1a8-de4e48eda7dc)

### SGD (Stochastic Gradient Descent)
- Updates model parameters using gradients from individual batches.
- A simple but effective optimizer for many tasks.

![SGDOptimizer](https://github.com/user-attachments/assets/b0fbb52c-35b7-49e4-b81c-45650ff40dd0)


### SGD with Momentum
- Incorporates a "momentum" term to SGD.
- Speeds up convergence by reducing oscillations.

![SGDwithMomentum](https://github.com/user-attachments/assets/056cb92a-8ae8-4942-a721-d28884a08c78)

### Adagrad (Adaptive Gradient Algorithm)
- Adapts learning rates based on historical gradient information.
- Particularly useful for sparse data or parameters.

### RMSprop (Root Mean Square Propagation)
- Similar to Adagrad but incorporates a decay factor to control learning rate adjustments.
- Addresses the issue of diminishing learning rates in Adagrad.

![RMSprop](https://github.com/user-attachments/assets/5adedf99-2702-4ced-8d15-a692015cd1a2)

## Conclusion
This repository emphasizes the role of optimizers in training neural networks. By understanding their strengths and weaknesses, users can make informed decisions about which optimizer to use for different tasks.

## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Contributions
Contributions are welcome! Feel free to fork the repository, improve the code, or add new optimizers to enhance the comparison. Submit pull requests with detailed descriptions of your changes.
