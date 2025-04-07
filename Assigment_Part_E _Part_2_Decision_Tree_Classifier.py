# Decision Tree Classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load dataset and remove duplicate rows
df = pd.read_csv('winequality-red.csv', sep=';')
df = df.drop_duplicates()

# Define features and target
X = df.drop('quality', axis=1)
y = df['quality']

# Split dataset into training (70%) and testing (30%), stratified by quality
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Hyperparameters
dt_classifier = DecisionTreeClassifier(
    max_depth=20,             # Maximum depth of the tree
    min_samples_split=2,     # Minimum samples required to split an internal node
    min_samples_leaf=1,      # Minimum samples required to be at a leaf node
    criterion='gini',        # Criterion used to measure the quality of a split ('gini' or 'entropy')
    random_state=42          # Ensures reproducibility of results
)
dt_classifier.fit(X_train, y_train)

# Predict on test set
y_pred_dt = dt_classifier.predict(X_test)

# Evaluate performance
accuracy_dt = accuracy_score(y_test, y_pred_dt)
conf_matrix_dt = confusion_matrix(y_test, y_pred_dt)
report_dt = classification_report(y_test, y_pred_dt)

print("Decision Tree Classifier Results:")
print("Accuracy:", accuracy_dt)
print("Confusion Matrix:\n", conf_matrix_dt)
print("Classification Report:\n", report_dt)
