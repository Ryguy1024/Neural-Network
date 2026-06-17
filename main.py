import numpy as np
import matplotlib.pyplot as plt
import urllib.request
import gzip

def load_mnist():
    def load_images(filename):
        with gzip.open(filename, 'rb') as f:
            raw = np.frombuffer(f.read(), np.uint8)
            return raw[16:].reshape(-1, 784) / 255.0

    def load_labels(filename):
        with gzip.open(filename, 'rb') as f:
            raw = np.frombuffer(f.read(), np.uint8)
            return raw[8:].astype(int)

    X_train = load_images("train-images-idx3-ubyte.gz")
    y_train = load_labels("train-labels-idx1-ubyte.gz")
    X_test  = load_images("t10k-images-idx3-ubyte.gz")
    y_test  = load_labels("t10k-labels-idx1-ubyte.gz")
    return X_train, y_train, X_test, y_test

X_train, y_train, X_test, y_test = load_mnist()
print(X_train.shape)  # (60000, 784)
print(X_test.shape)   # (10000, 784)

# Network dimensions
input_size = 784      # 8x8 pixels
hidden_size = 128     # neurons in the hidden layer
output_size = 10     # digits 0-9

# Initialise weights and biases randomly
np.random.seed(42)
W1 = np.random.randn(input_size, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, output_size) * 0.1
b2 = np.zeros((1, output_size))

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def one_hot(y, num_classes=10):
    result = np.zeros((len(y), num_classes))
    result[np.arange(len(y)), y] = 1
    return result

def loss(a2, y):
    y_hot = one_hot(y)
    return np.mean((a2 - y_hot) ** 2)

def relu(z):
    return np.maximum(0, z)

def forward(X):
    z1 = X @ W1 + b1
    a1 = relu(z1)              # ReLU in hidden layer
    z2 = a1 @ W2 + b2
    a2 = sigmoid(z2)           # keep sigmoid on output
    return z1, a1, z2, a2

def backward(X, y, z1, a1, z2, a2, learning_rate=0.1):
    global W1, b1, W2, b2

    m = len(X)
    y_hot = one_hot(y)

    dL_da2 = 2 * (a2 - y_hot) / m
    da2_dz2 = a2 * (1 - a2)
    dz2 = dL_da2 * da2_dz2

    dL_da1 = dz2 @ W2.T
    da1_dz1 = (z1 > 0).astype(float)   # ReLU derivative
    dz1 = dL_da1 * da1_dz1

    W2 -= a1.T @ dz2 * learning_rate
    b2 -= np.sum(dz2, axis=0, keepdims=True) * learning_rate
    W1 -= X.T @ dz1 * learning_rate
    b1 -= np.sum(dz1, axis=0, keepdims=True) * learning_rate

def predict(X):
    _, _, _, a2 = forward(X)
    return np.argmax(a2, axis=1)  # pick the highest output neuron

# Training loop
# Use a subset of the data
X_train_small = X_train[:10000]
y_train_small = y_train[:10000]

losses = []
accuracies = []

epochs = 1000
for epoch in range(epochs):
    z1, a1, z2, a2 = forward(X_train_small)
    backward(X_train_small, y_train_small, z1, a1, z2, a2)

    if epoch % 50 == 0:
        current_loss = loss(a2, y_train_small)
        accuracy = np.mean(predict(X_test) == y_test) * 100
        losses.append(current_loss)
        accuracies.append(accuracy)
        print(f"Epoch {epoch} — Loss: {current_loss:.4f} — Test accuracy: {accuracy:.1f}%")

# Plot 2 — training progress
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.plot(losses, color='crimson')
ax1.set_title('Loss over time')
ax1.set_xlabel('Epoch (x50)')
ax1.set_ylabel('Loss')

ax2.plot(accuracies, color='steelblue')
ax2.set_title('Accuracy over time')
ax2.set_xlabel('Epoch (x50)')
ax2.set_ylabel('Accuracy %')

plt.tight_layout()
plt.show()

# Plot 1 — what the hidden neurons have learned
fig, axes = plt.subplots(8, 16, figsize=(16, 8))
for i, ax in enumerate(axes.flat):
    ax.imshow(W1[:, i].reshape(28, 28), cmap='RdBu')
    ax.axis('off')
plt.suptitle('What each hidden neuron has learned to look for')
plt.tight_layout()
plt.show()

# Show some predictions
fig, axes = plt.subplots(2, 5, figsize=(10, 4))
for i, ax in enumerate(axes.flat):
    ax.imshow(X_test[i].reshape(28, 28), cmap='gray')
    pred = predict(X_test[i:i+1])[0]
    actual = y_test[i]
    color = 'green' if pred == actual else 'red'
    ax.set_title(f'Pred: {pred} / Act: {actual}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()