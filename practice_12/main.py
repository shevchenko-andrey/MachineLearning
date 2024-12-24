import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Step 1: Read the CSV file and display the first 5 records
df = pd.read_csv('cardio.csv', delimiter=';')
print(df.head())

# Step 2: Use only the first 30 records for this task
df = df.head(30)

# Step 3: Define the feature matrix X and target vector y
X = df[['age', 'gender', 'height', 'weight', 'ap_hi', 'ap_lo',
        'cholesterol', 'gluc', 'smoke', 'alco', 'active']]  # Features
y = df['cardio']  # Target variable (cardio disease presence)

# Step 4: Split the data into training and test sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Create and train the decision tree classifier model
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 6: Visualize the decision tree
plt.figure(figsize=(15, 10))
plot_tree(model, feature_names=X.columns, filled=True)
plt.show()

# Step 7: Predict the outcomes on the test data
y_pred = model.predict(X_test)

# Step 8: Print the classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Step 9: Compute and visualize the confusion matrix
cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot()
plt.show()
