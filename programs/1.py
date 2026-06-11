import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def relu(x):
    return np.maximum(0, x)

def leaky_relu(x, alpha=0.1):
    return np.where(x > 0, x, alpha * x)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

x = np.linspace(-5, 5, 100)

plt.figure(figsize=(12, 8))

plt.subplot(2, 3, 1)
plt.plot(x, sigmoid(x)); plt.title("Sigmoid"); plt.grid(True)

plt.subplot(2, 3, 2)
plt.plot(x, tanh(x)); plt.title("Tanh"); plt.grid(True)

plt.subplot(2, 3, 3)
plt.plot(x, relu(x)); plt.title("ReLU"); plt.grid(True)

plt.subplot(2, 3, 4)
plt.plot(x, leaky_relu(x)); plt.title("Leaky ReLU"); plt.grid(True)

plt.subplot(2, 3, 5)
plt.plot(x, softmax(x)); plt.title("Softmax"); plt.grid(True)

plt.tight_layout()
plt.show()

sample = np.array([1, 2, 3])
probabilities = softmax(sample)
plt.bar(["Class A", "Class B", "Class C"], probabilities)
plt.title("Softmax Output")
plt.ylabel("Probability")
plt.show()