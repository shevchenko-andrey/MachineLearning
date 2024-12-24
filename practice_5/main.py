import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression

# Task #1: Custom implementation of The Least Squares Method
def linear_least_squares(x, y):
    m_x = np.mean(x)
    m_y = np.mean(y)

    alpha_1 = np.mean(x * y)
    alpha_2 = np.mean(x * x)

    omega_1 = (alpha_1 - m_x * m_y) / (alpha_2 - m_x ** 2)
    omega_0 = m_y - omega_1 * m_x

    return omega_1, omega_0

# Task #1: Generate synthetic data using make_regression
X, Y = make_regression(n_samples=300, n_features=1, n_informative=1, noise=5, random_state=10)

# Compute parameters using the custom method (The Least Squares Method)
omega_1, omega_0 = linear_least_squares(X.flatten(), Y)

print(f"Task 1 - Custom Model ω1: {omega_1}")
print(f"Task 1 - Custom Model ω0: {omega_0}")

# Task #2: Apply LinearRegression from scikit-learn
model = LinearRegression()
model.fit(X, Y)

# Get the coefficients from the sklearn model
omega_1_sklearn = model.coef_[0]
omega_0_sklearn = model.intercept_

print(f"Task 2 - LinearRegression ω1: {omega_1_sklearn}")
print(f"Task 2 - LinearRegression ω0: {omega_0_sklearn}")

plt.scatter(X, Y, label='Data', color='red')

# Line for custom model (least squares method)
x_line = np.linspace(X.min(), X.max(), 100)
y_line_custom = omega_1 * x_line + omega_0
plt.plot(x_line, y_line_custom, label=f'Custom Model: $f(x) = {omega_1:.2f}x + {omega_0:.2f}$', color='blue')

# Line for LinearRegression model
y_line_sklearn = omega_1_sklearn * x_line + omega_0_sklearn
plt.plot(x_line, y_line_sklearn, label=f'LinearRegression: $f(x) = {omega_1_sklearn:.2f}x + {omega_0_sklearn:.2f}$',
         color='green')

# Labeling the axes
plt.xlabel('Feature')
plt.ylabel('Target')

# Display the legend
plt.legend()

# Show the plot
plt.show()
