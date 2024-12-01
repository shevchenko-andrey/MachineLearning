import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Generate data
N = 100
x = np.random.uniform(0, 1, N)
y = np.sin(2 * np.pi * x) + np.random.normal(0, 0.2, N)

# Step 2: Split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 3: Initialize the degrees for polynomial features
degrees = [1, 2, 3, 6]

# Step 4: Create and train the models for each degree
models = {}
for degree in degrees:
    # Generate polynomial features for training data
    poly = PolynomialFeatures(degree)
    x_train_poly = poly.fit_transform(x_train.reshape(-1, 1))

    # Train a linear regression model
    model = LinearRegression()
    model.fit(x_train_poly, y_train)
    models[degree] = model  # Store the model

# Step 5: Test the models on the test data and calculate performance metrics
results = {}
for degree in degrees:
    poly = PolynomialFeatures(degree)
    x_test_poly = poly.fit_transform(x_test.reshape(-1, 1))

    y_pred = models[degree].predict(x_test_poly)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    results[degree] = {'RMSE': rmse, 'R2': r2}

for degree, metrics in results.items():
    print(f"Degree {degree}: RMSE = {metrics['RMSE']:.4f}, R^2 = {metrics['R2']:.4f}")

plt.figure(figsize=(10, 6))
x_range = np.linspace(0, 1, 1000)
for degree in degrees:
    poly = PolynomialFeatures(degree)
    x_range_poly = poly.fit_transform(x_range.reshape(-1, 1))

    y_range_pred = models[degree].predict(x_range_poly)

    plt.plot(x_range, y_range_pred, label=f'Degree {degree}')

plt.scatter(x, y, color='black', s=10, label='Data')
plt.title("Polynomial Regression (Various Degrees)")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()
