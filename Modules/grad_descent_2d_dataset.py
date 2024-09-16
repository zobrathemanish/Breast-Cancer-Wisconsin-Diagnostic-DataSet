import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic dataset
np.random.seed(0)
X = 2 * np.random.rand(100, 1)
true_w0 = np.random.randn()
true_w1 = np.random.randn()
y = true_w0 + true_w1 * X + 0.25 * np.random.randn(100, 1)

# Plot the dataset
plt.figure(figsize=(10, 5))
plt.scatter(X, y)
plt.xlabel('x')
plt.ylabel('y')
plt.title('Data Scatter')
plt.grid()
plt.show()

# Add a bias term (column of ones) to the input data
X_b = np.c_[np.ones((100, 1)), X]

# Define the loss function (mean squared error)
def loss_function(w, X_b, y):
    m = len(y)
    predictions = X_b.dot(w)
    return (1 / (2 * m)) * np.sum((predictions - y) ** 2)

# Define the gradient of the loss function
def gradient(w, X_b, y):
    m = len(y)
    predictions = X_b.dot(w)
    return (1 / m) * X_b.T.dot(predictions - y)

# Gradient descent parameters
learning_rate = 0.1
num_iterations = 300

# Initialize weights within a reasonable range
w = np.random.randn(2, 1)  # Starting point
w_history = [w.copy()]  # To store the history of weight values
loss_history = [loss_function(w, X_b, y)]  # To store the history of loss values

# Gradient descent loop
for i in range(num_iterations):
    grad = gradient(w, X_b, y)
    w = w - learning_rate * grad
    w_history.append(w.copy())
    loss_history.append(loss_function(w, X_b, y))

# Convert history lists to arrays for easier plotting
w_history = np.array(w_history)
loss_history = np.array(loss_history)

# Plot the loss over iterations
plt.figure(figsize=(10, 5))
plt.plot(range(num_iterations + 1), loss_history, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss over Iterations')
plt.legend()
plt.grid()
plt.show()

# Plot the dataset and the fitted line
plt.figure(figsize=(10, 5))
plt.scatter(X, y, label='Data')
plt.plot(X, X_b.dot(w), color='red', label='Fitted line')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Linear Regression Fit')
plt.legend()
plt.grid()
plt.show()

# 3D plot of the loss function surface and gradient descent path
fig = plt.figure(figsize=(14, 6))

# Loss function surface plot
ax1 = fig.add_subplot(121, projection='3d')
w0_values = np.linspace(np.min(w_history[:, 0]) - 1, np.max(w_history[:, 0]) + 1, 100)
w1_values = np.linspace(np.min(w_history[:, 1]) - 1, np.max(w_history[:, 1]) + 1, 100)
W0, W1 = np.meshgrid(w0_values, w1_values)
Z = np.array([[loss_function(np.array([[w0], [w1]]), X_b, y) for w0 in w0_values] for w1 in w1_values])
ax1.plot_surface(W0, W1, Z, alpha=0.6, cmap='viridis')

# Gradient descent path
ax1.plot(w_history[:, 0], w_history[:, 1], loss_history, color='red', marker='o', label='Gradient Descent Path')
ax1.set_xlabel('w0')
ax1.set_ylabel('w1')
ax1.set_zlabel('Loss')
ax1.set_title('Gradient Descent on Loss Function')
ax1.legend()

# 2D contour plot
ax2 = fig.add_subplot(122)
contour = ax2.contour(W0, W1, Z, levels=50, cmap='viridis')
ax2.plot(w_history[:, 0], w_history[:, 1], color='red', marker='o', label='Gradient Descent Path')
ax2.set_xlabel('w0')
ax2.set_ylabel('w1')
ax2.set_title('2D Projection of Gradient Descent')
ax2.legend()
plt.colorbar(contour, ax=ax2, shrink=0.8)

plt.show()

# Print the final value of weights and the corresponding loss
print(f"True weights: {[true_w0, true_w1]}")
print(f"Final weights: {w.ravel()}")
print(f"Final loss: {loss_function(w, X_b, y)}")
