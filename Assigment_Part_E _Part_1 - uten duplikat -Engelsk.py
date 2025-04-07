import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

# Read the dataset. File separator is a semicolon.
df = pd.read_csv('winequality-red.csv', sep=';')

# Check for duplicate rows
print("\nNumber of duplicate rows:")
print(df.duplicated().sum())

# Drop duplicate rows
df = df.drop_duplicates()
print("\nNumber of rows after dropping duplicates:", df.shape[0])

# Print the first rows to inspect the data structure
print("The first rows of the dataset:")
print(df.head())

# Check data structure and missing values
print("\nData structure:")
print(df.info())

print("\nNumber of missing values per column:")
print(df.isnull().sum())

# Statistical summary
print("\nStatistical summary:")
print(df.describe())

# Print median for each column (median is also shown as 50% in describe())
print("\nMedian for each column:")
print(df.median())

# Compute the mode (note: .mode() can return multiple values, we take the first one)
mode_values = df.mode().iloc[0]
print("\nMode for each column:")
print(mode_values)

# Compute first and third quartile (Q1 and Q3)
q1 = df.quantile(0.25)
q3 = df.quantile(0.75)
print("\nFirst quartile (Q1):")
print(q1)
print("\nThird quartile (Q3):")
print(q3)

# Plot 1: Histogram for each feature
df.hist(bins=20, figsize=(12,10))
plt.tight_layout()
plt.show()

# Plot 2: Boxplot for each feature (used to identify outliers)
plt.figure(figsize=(12,8))
for i, column in enumerate(df.columns, 1):
    plt.subplot(3, 4, i)
    sns.boxplot(y=df[column])
    plt.title(column)
plt.tight_layout()
plt.show()

# Plot 3: Correlation matrix with heatmap
plt.figure(figsize=(10,8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.title("Correlation Matrix")
plt.show()

# Plot 4: Scatter plot for selected features, alcohol vs. quality
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='alcohol', y='quality', hue='quality')
plt.title("Scatter plot: Alcohol vs Quality")
plt.show()

# Plot 5: Pairplot to see relationships between all features
sns.pairplot(df, diag_kind='kde', corner=True)
plt.show()

# Plot 6: Histogram for quality
plt.figure(figsize=(6,4))
sns.countplot(x='quality', data=df, palette='viridis')
plt.title("Distribution of Wine Quality Scores")
plt.xlabel("Quality Score")
plt.ylabel("Frequency")
plt.show()

# Plot 7: Scatter plot for selected features, volatile acidity vs. quality
plt.figure(figsize=(8,6))
sns.scatterplot(data=df, x='volatile acidity', y='quality', hue='quality')
plt.title("Scatter plot: volatile acidity vs Quality")
plt.show()

# Extra: Identification of outliers using Z-score 
z_scores = np.abs(stats.zscore(df.select_dtypes(include=[np.number])))
outliers = (z_scores > 3).sum(axis=0)
print("\nNumber of outliers per feature (Z-score > 3):")


print(outliers)

# Check the distribution of the quality variable
quality_counts = df['quality'].value_counts()
print("Distribution of wine quality scores:")
print(quality_counts)


