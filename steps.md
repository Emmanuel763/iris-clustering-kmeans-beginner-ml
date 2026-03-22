# 🌸 Iris Species Clustering: A Step-by-Step Guide

This guide walks through the process of unsupervised learning using the **K-Means Clustering** algorithm on the famous Iris dataset.

---

## 📑 Table of Contents
1. [Import Libraries](#step-1-import-libraries)
2. [Load Dataset](#step-2-load-dataset)
3. [Explore Data](#step-3-explore-data)
4. [Data Visualization](#step-4-data-visualization)
5. [K-Means & Elbow Method](#step-5-apply-k-means)
6. [Model Evaluation](#step-6-evaluate-model)

---

## 📦 Step 1: Import Libraries
We begin by importing the necessary tools for data manipulation, visualization, and machine learning.

```python
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
```

## Explaination
- pandas (pd) → work with table data  
- load_iris → get example dataset  
- seaborn (sns) → make nice graphs  
- matplotlib.pyplot (plt) → draw charts  
- KMeans → group similar data  
- silhouette_score → check grouping quality  

---

# 🟢 `Step 2: Load Dataset`

We load the built-in Iris dataset and convert it into a structured format for easier analysis.

---

## 📌 Code

```python
from sklearn.datasets import load_iris
import pandas as pd

# Load dataset
iris = load_iris()

# Convert to DataFrame
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Add target column
df['species'] = iris.target

# Preview dataset
df.head()
```
## 🖼️ Output Placeholder
---

# 🟢 `Step 3: Explore Data`

```md

We inspect and understand the dataset.
```

## Code

```python
# Show the keys in the Iris dataset
print("Keys:", iris.keys())

# Show target values (the species labels as numbers)
print("\nTarget Values:", iris.target)

# Show target names (species names)
print("\nTarget Names:", iris.target_names)

# Create a DataFrame for easier inspection
df = pd.DataFrame(iris.data, columns=iris.feature_names)

# Display the first 5 rows
df.head()
```

## Extra
```python
# Show basic info: column names, data types, missing values
df.info()

# Show statistical summary: mean, min, max, etc.
df.describe()
```


---

# 🟢 `Step 4: Data Visualization

```md

We visualize relationships between features.
```

## Code

```python
sns.pairplot(df)
plt.show()
```
## 🖼️ Output Placeholder

---

# 🟢 `Step 5: Apply K-Means`

```md

We use the Elbow Method to find optimal clusters.
```
## Code

```python
X = df[['petal length (cm)', 'petal width (cm)']]

error = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    error.append(kmeans.inertia_)

plt.plot(range(1, 11), error)
plt.show()
```
## 🖼️ Output Placeholder
---

# 🟢 `Step 6: Evaluate Model`

```md
# 📏 

We apply K-Means and evaluate performance.
```
## Code

```python
kmeans = KMeans(n_clusters=3, init='k-means++', random_state=42)
y_predict = kmeans.fit_predict(X)

plt.scatter(X.iloc[:,0], X.iloc[:,1], c=y_predict)
plt.show()

score = silhouette_score(X, y_predict)
print("Silhouette Score:", score)
```
## 🖼️ Output Placeholder