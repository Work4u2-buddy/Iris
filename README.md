# Iris
🌸 Hands-on ML project using the Iris dataset. Includes data preprocessing, model training (Linear Regression), accuracy evaluation, and visualizations. Perfect for beginners exploring supervised learning.


🌸 **My First Machine Learning Project with the Iris Dataset!** 🌸

I recently trained my very first ML model using the famous **Iris dataset** 🌱.
Here’s a step-by-step breakdown of the journey:

🔹 **Step 1: Import libraries**

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
```

🔹 **Step 2: Load dataset**

```python
from sklearn.datasets import load_iris
iris = load_iris()
```

🔹 **Step 3: Create a DataFrame**

```python
df = pd.DataFrame(iris.data, columns=iris.feature_names)
df.head()
```

👉 `head()` shows the first few rows of the dataset.

🔹 **Step 4: Add target column**

```python
df['target'] = iris.target
```

➡️ If you want to add rows instead of columns, you can use `df.append()` or `df.loc[]`.

🔹 **Step 5: Select features (X) and target (y)**

```python
X = df.drop(['species', 'target'], axis=1)   # Features  
y = df['target']                             # Labels  
```

🔹 **Step 6: Split into Training & Testing sets**

```python
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

🔹 **Step 7: Train the model**

```python
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train, y_train)
print("Model trained successfully!")
```

⚠️ **Learning Moment:**
At first, I mistakenly used text labels (like “setosa”, “versicolor”, “virginica”) as targets, which caused errors 🚨.
Solution ➝ Always use numeric labels for training ML models.

🔹 **Step 8: Evaluate the model**
✔️ Calculate **accuracy score**
✔️ Validate results with a **confusion matrix**
✔️ Plot graphs for better visualization 📊

---

✨ This project taught me:
✅ How to handle datasets in Pandas
✅ How to split data into training/testing sets
✅ The importance of choosing correct target values
✅ The full ML workflow from **data → model → evaluation**

Excited to keep building more ML projects! 🚀

