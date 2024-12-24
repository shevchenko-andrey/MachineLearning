import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv('hearing_test.csv')

# Display the first 5 rows of the dataset
print(df.head())

# Show the count of passed and failed tests
print(df['test_result'].value_counts())

# Scatter plot for physical_score vs test_result
plt.figure(figsize=(10, 5))
plt.scatter(x='physical_score', y='test_result', data=df, alpha=0.6, c=df['test_result'], cmap='coolwarm')
plt.xlabel('Physical Score')
plt.ylabel('Test Result')
plt.title('Scatter Plot: Physical Score vs Test Result')
plt.colorbar(label='Test Result')
plt.show()

# Scatter plot for age vs test_result
plt.figure(figsize=(10, 5))
plt.scatter(x='age', y='test_result', data=df, alpha=0.6, c=df['test_result'], cmap='coolwarm')
plt.xlabel('Age')
plt.ylabel('Test Result')
plt.title('Scatter Plot: Age vs Test Result')
plt.colorbar(label='Test Result')
plt.show()

# 3D scatter plot for age, physical_score, and test_result
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Scatter plot in 3D space
scatter = ax.scatter(df['age'], df['physical_score'], df['test_result'], c=df['test_result'], cmap='coolwarm')

ax.set_xlabel('Age')
ax.set_ylabel('Physical Score')
ax.set_zlabel('Test Result')

ax.set_title('3D Scatter Plot: Age, Physical Score vs Test Result')

fig.colorbar(scatter, label='Test Result')

plt.show()

# Feature matrix X and target vector y
X = df[['age', 'physical_score']]  # Select the features (age, physical_score)
y = df['test_result']  # Select the target (test_result)

# Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_test = scaler.transform(X_test)

# Initialize and train the logistic regression model
model = LogisticRegression()
model.fit(scaled_X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(scaled_X_test)

# Calculate and display the classification accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Classification Accuracy: {accuracy:.4f}")
