# Neural Network from Scratch

A handwritten digit classifier built from scratch using only NumPy. No ML libraries — just maths.

Trained on the MNIST dataset (70,000 handwritten digits), achieving ~90% accuracy on unseen test data.

Now includes a Flask API and interactive canvas frontend so you can draw a digit and get a live prediction.

## How it works

- **Forward pass** — multiplies inputs through two layers of weights to produce a prediction
- **Backpropagation** — calculates how wrong the prediction was and nudges weights in the right direction
- **ReLU activation** — used in the hidden layer to avoid vanishing gradients
- **Sigmoid activation** — used on the output layer to produce values between 0 and 1

## Architecture

```
Input (784)  →  Hidden (128, ReLU)  →  Output (10, Sigmoid)
```

## Results

- Test accuracy: ~90.7%
- Trained on 10,000 samples for speed
- 1,000 epochs

## Project Structure

```
├── main.py              # Train the network and save weights to weights.npz
├── app.py               # Flask API — loads weights and exposes POST /predict
├── templates/
│   └── index.html       # Canvas UI for drawing digits and viewing predictions
└── *.gz                 # MNIST data files
```

## Setup & Usage

Install dependencies:
```bash
pip install numpy matplotlib flask pillow
```

**Step 1 — Train and save weights:**
```bash
python main.py
```
This trains the network and writes `weights.npz`. Only needs to be run once.

**Step 2 — Start the server:**
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000), draw a digit on the canvas, and get a live prediction with per-digit confidence scores.

## What I learned

Built as a first neural network project to understand the fundamentals of how networks learn — 
without hiding the maths behind a library like PyTorch or TensorFlow.
