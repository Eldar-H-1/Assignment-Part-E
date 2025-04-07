# Random Forest Classifier
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
rf_classifier = RandomForestClassifier(
    n_estimators=100,        # Number of trees in the forest
    min_samples_split=2,     # Minimum samples required to split an internal node
    min_samples_leaf=1,      # Minimum samples required in a leaf node
    max_depth=2,             # Maximum depth of each tree
    criterion='gini',        # Criterion used to measure the quality of a split ('gini' or 'entropy')
    random_state=42          # Ensures reproducibility of results
)
rf_classifier.fit(X_train, y_train)

# Predict on test set
y_pred_rf = rf_classifier.predict(X_test)

# Evaluate performance
accuracy_rf = accuracy_score(y_test, y_pred_rf)
conf_matrix_rf = confusion_matrix(y_test, y_pred_rf)
report_rf = classification_report(y_test, y_pred_rf)

print("Random Forest Classifier Results:")
print("Accuracy:", accuracy_rf)
print("Confusion Matrix:\n", conf_matrix_rf)
print("Classification Report:\n", report_rf)
