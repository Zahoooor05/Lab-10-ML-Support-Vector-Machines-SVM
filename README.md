# Lab-10-ML-Support-Vector-Machines-SVM

## Overview
This project demonstrates the implementation of a Support Vector Machine (SVM) classifier using the famous Iris Flower Dataset. The assignment includes data exploration, visualization, model training, evaluation, and hyperparameter tuning using GridSearchCV in Python.

---

## Dataset Information
The Iris dataset contains 150 flower samples from three species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

### Features
Each sample includes four features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The dataset is loaded using Seaborn:

```python
iris = sns.load_dataset('iris')
```

---

## Project Objectives
The objectives of this assignment are:

- Load and explore the Iris dataset
- Visualize feature relationships
- Train an SVM classifier
- Evaluate model performance
- Tune hyperparameters using GridSearchCV
- Compare results before and after optimization

---

## Libraries Used

```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV
```

---

## Exploratory Data Analysis (EDA)

### Pairplot Visualization
A pairplot was created to visualize relationships between all features and identify class separability.

### KDE Plot
A KDE (Kernel Density Estimation) plot was generated for:
- Sepal Length
- Sepal Width

for the Setosa species.

---

## Train-Test Split

The dataset was divided into:
- 75% Training Data
- 25% Testing Data

```python
X = iris.drop(columns=['species'])
y = iris['species']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
```

---

## SVM Model Training

```python
model = SVC()
model.fit(X_train, y_train)
```

---

## Model Evaluation

Predictions were generated using:

```python
y_pred = model.predict(X_test)
```

The model was evaluated using:
- Confusion Matrix
- Classification Report

Initial model accuracy achieved approximately 95%.

---

## Hyperparameter Tuning with GridSearchCV

### Parameter Grid

```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001]
}
```

### GridSearchCV Implementation

```python
grid = GridSearchCV(SVC(), param_grid, refit=True, verbose=2)
grid.fit(X_train, y_train)
```

### Best Parameters

```python
{'C': 1, 'gamma': 1}
```

---

## Final Results

The optimized SVM model maintained high classification performance with approximately 95% accuracy on the test dataset.

---

## Conclusion

This assignment successfully demonstrated the use of Support Vector Machines (SVM) for multiclass classification using the Iris dataset. The project included:
- Data visualization
- Model training
- Performance evaluation
- Hyperparameter tuning

The results showed that SVM is highly effective for small and well-structured datasets.

---

## Tools & Technologies
- Python
- Scikit-learn
- Seaborn
- Matplotlib
- Google Colab 
