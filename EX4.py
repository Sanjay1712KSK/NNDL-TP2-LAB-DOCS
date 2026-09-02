import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Training data
# -----------------------------
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dtype=float)
y = np.array([2, 4, 5, 8, 10, 12, 14, 16, 18, 20], dtype=float)

# -----------------------------
# 2. Initialize parameters
# -----------------------------
w = 0.0
b = 0.0

learning_rate = 0.01
epochs = 100

# -----------------------------
# 3. Stochastic Gradient Descent
# -----------------------------
for epoch in range(epochs):

    # Shuffle data
    indices = np.random.permutation(len(X))

    for i in indices:

        # Select ONE data point
        x_i = X[i]
        y_i = y[i]

        # Prediction
        y_pred = w * x_i + b

        # Error
        error = y_pred - y_i

        # Gradients
        dw = error * x_i
        db = error

        # Update weights
        w = w - learning_rate * dw
        b = b - learning_rate * db

# -----------------------------
# 4. Final predictions
# -----------------------------
y_pred = w * X + b

print("Weight (w):", w)
print("Bias (b):", b)

# -----------------------------
# 5. Plot graph
# -----------------------------
plt.scatter(X, y, color="blue", label="Actual Data")

plt.plot(
    X,
    y_pred,
    color="red",
    linewidth=2,
    label="SGD Regression Line"
)

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression using Stochastic Gradient Descent")

plt.legend()
plt.grid(True)

plt.show()