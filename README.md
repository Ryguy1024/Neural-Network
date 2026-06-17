# Neural Network from Scratch

A handwritten digit classifier built from scratch using only NumPy. No ML libraries — just maths.

Trained on the MNIST dataset (70,000 handwritten digits), achieving ~90% accuracy on unseen test data.

## How it works

- **Forward pass** — multiplies inputs through two layers of weights to produce a prediction
- **Backpropagation** — calculates how wrong the prediction was and nudges weights in the right direction
- **ReLU activation** — used in the hidden layer to avoid vanishing gradients
- **Sigmoid activation** — used on the output layer to produce values between 0 and 1

## Results

- Test accuracy: ~90.7%
- Trained on 10,000 samples for speed
- 1000 epochs

## Setup

Install dependencies:
```bash
pip install numpy matplotlib scikit-learn
```

Run the network:
```bash
python main.py
```

## What I learned

Built as a first neural network project to understand the fundamentals of how networks learn — 
without hiding the maths behind a library like PyTorch or TensorFlow.
