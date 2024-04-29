---
jupytext:
  cell_metadata_filter: all, -hidden, -heading_collapsed, -run_control, -trusted
  notebook_metadata_filter: all, -jupytext.text_representation.jupytext_version, -jupytext.text_representation.format_version,
    -language_info.version, -language_info.codemirror_mode.version, -language_info.codemirror_mode,
    -language_info.file_extension, -language_info.mimetype, -toc
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
language_info:
  name: python
  nbconvert_exporter: python
  pygments_lexer: ipython3
nbhosting:
  title: Introduction Statistical Learning
  version: '1.0'
---

<div class="licence">
<span><img src="media/logo_IPParis.png" /></span>
<span>Aymeric DIEULEVEUT</span>
<span>Licence CC BY-NC-ND</span>
</div>

+++

# Lab 1: Binary Classification on MNIST Dataset

In this Jupyter Notebook, the goal is to explore three different classification methods to distinguish between odd and even digits from the MNIST dataset using:

-    Linear Discriminant Analysis (LDA)
-    Quadratic Discriminant Analysis (QDA)
-    Logistic Regression

The MNIST dataset consists of handwritten digits, each being a 28x28 pixel grayscale image, that were mentioned in the lecture. For the purpose of this exercise, the digits will be divided into two classes:

    Even (0, 2, 4, 6, 8)
    Odd (1, 3, 5, 7, 9)

This is meant to have a binary classification task

You will preprocess the data, apply each of the three classifiers, and assess their performance.


#### Setup and Data Preprocessing

First, we will load the necessary libraries and the MNIST dataset. We will then preprocess the data to be suitable for classification.


```{code-cell} python
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# Load MNIST dataset
digits = datasets.load_digits()

# Display the first few images and labels
fig, axes = plt.subplots(1, 4, figsize=(10, 3))
for ax, image, label in zip(axes, digits.images, digits.target):
    ax.set_axis_off()
    ax.imshow(image, cmap=plt.cm.gray_r, interpolation='nearest')
    ax.set_title('Label: %i' % label)
plt.show()

# Define a function to convert labels to even (0) and odd (1)
def convert_labels(labels):
    return np.array([0 if label % 2 == 0 else 1 for label in labels])

# Apply the function to the labels
binary_labels = convert_labels(digits.target)

# Flatten the images
n_samples = len(digits.images)
data = digits.images.reshape((n_samples, -1))


```


```{code-cell} python
# Splitting data
# Split data into training and testing sets


X_train, X_test, y_train, y_test = # YOUR CODE OR ANSWER HERE
```

## Question 1:

What are the dimensions of X_train and X_test? How many features does each sample have?
Check the proportion of each label in the train and text datasets

#### Linear Discriminant Analysis (LDA)

Now, apply LDA on the training data and evaluate its performance.


```{code-cell} python
# Initialize and fit LDA
lda = # YOUR CODE OR ANSWER HERE

# Fit the model 
# YOUR CODE OR ANSWER HERE

# Predict on the test set
y_pred_lda = # YOUR CODE OR ANSWER HERE

# Evaluate LDA
print("LDA Accuracy: ", accuracy_score(y_test, y_pred_lda))
print("LDA Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lda))
print("LDA Classification Report:\n", classification_report(y_test, y_pred_lda))
```

## Question 2:

Based on the confusion matrix and classification report, how does LDA perform in distinguishing between odd and even digits?
Quadratic Discriminant Analysis (QDA)

Next, apply QDA and evaluate its performance.


```{code-cell} python
# Initialize and fit QDA
qda = # YOUR CODE OR ANSWER HERE
# YOUR CODE OR ANSWER HERE

# Predict on the test set
y_pred_qda = # YOUR CODE OR ANSWER HERE

# Evaluate QDA
print("QDA Accuracy: ", accuracy_score(y_test, y_pred_qda))
print("QDA Confusion Matrix:\n", confusion_matrix(y_test, y_pred_qda))
print("QDA Classification Report:\n", classification_report(y_test, y_pred_qda))
```

## Question 3:

Compare the performance of QDA to LDA. Which one performs better and why might that be?
Logistic Regression

Lastly, apply Logistic Regression and assess its performance.


```{code-cell} python
# Initialize and fit Logistic Regression
log_reg = # YOUR CODE OR ANSWER HERE
# YOUR CODE OR ANSWER HERE

# Predict on the test set
y_pred_log_reg = # YOUR CODE OR ANSWER HERE

# Evaluate Logistic Regression
print("Logistic Regression Accuracy: ", accuracy_score(y_test, y_pred_log_reg))
print("Logistic Regression Confusion Matrix:\n", confusion_matrix(y_test, y_pred_log_reg))
print("Logistic Regression Classification Report:\n", classification_report(y_test, y_pred_log_reg))
```

## Question 4:

Discuss the performance of Logistic Regression compared to LDA and QDA. Which classifier would you recommend for this task and why?

#### Conclusion
You have now implemented three different classifiers on a binary classification problem using the MNIST dataset. 

## Question 5:

Reflect on the importance of preprocessing and the choice of model in the context of this task. How could further tuning or different preprocessing steps potentially impact the results?




```{code-cell} python
import matplotlib.pyplot as plt

# Define different training set sizes
training_sizes = [10, 100, 1000, len(X_train)]

# Store accuracies for each model
accuracies_lda = []
accuracies_qda = []
accuracies_log_reg = []

# Function to train model and calculate accuracy
def train_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    accuracy = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, y_pred_train)
    return (accuracy, train_acc)

# Evaluate models on different training sizes
for size in training_sizes:
    X_train_subset = # YOUR CODE OR ANSWER HERE
    y_train_subset = # YOUR CODE OR ANSWER HERE

    # Ensure there's at least one example of each class
    if len(np.unique(y_train_subset)) < 2:
        continue

    # LDA
    
    lda = # YOUR CODE OR ANSWER HERE
    acc_lda , _= # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

    # QDA
    qda = # YOUR CODE OR ANSWER HERE
    acc_qda , _ = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

    # Logistic Regression      
    log_reg = # YOUR CODE OR ANSWER HERE
    acc_log_reg , _ = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(training_sizes, accuracies_lda, label='LDA', marker='o')
plt.plot(training_sizes, accuracies_qda, label='QDA', marker='o')
plt.plot(training_sizes, accuracies_log_reg, label='Logistic Regression', marker='o')
plt.title('Accuracy vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

## Question 6:

From the plot, how does the accuracy of each model change as the number of training points increases? Which model appears to be the most sensitive to the size of the training set?

This plot and analysis will help you understand the behavior of each classification method as the amount of training data varies, providing insight into their robustness and efficiency in learning from limited data.

## Question 7:
How what so you think of the perfomance of QDA?



## Question 8: 
- Compute the train accuracy of the different methods and QDA.
- Is there any overfitting happpening?


```{code-cell} python
import matplotlib.pyplot as plt

# Define different training set sizes
training_sizes = [10, 100, 300, 700, 1000, len(X_train)]

# Store accuracies for each model
accuracies_lda = []
accuracies_qda = []
accuracies_log_reg = []

#Also Store TRAIN accuracies for each model
accuracies_lda_tr = []
accuracies_qda_tr = []
accuracies_log_reg_tr = []

# Evaluate models on different training sizes
for size in training_sizes:
    X_train_subset = # YOUR CODE OR ANSWER HERE
    y_train_subset = # YOUR CODE OR ANSWER HERE

    # Ensure there's at least one example of each class
    if len(np.unique(y_train_subset)) < 2:
        continue

    # LDA
    lda = # YOUR CODE OR ANSWER HERE
    acc_lda , acc_lda_tr= # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

    
    # QDA
    qda = # YOUR CODE OR ANSWER HERE
    acc_qda , acc_qda_tr = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    

    # Logistic Regression
    log_reg = # YOUR CODE OR ANSWER HERE
    acc_log_reg , acc_log_reg_tr = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(training_sizes, accuracies_lda, label='LDA', marker='o', c='b')
plt.plot(training_sizes, accuracies_qda, label='QDA', marker='o', c='g')
plt.plot(training_sizes, accuracies_log_reg, label='Logistic Regression', marker='o', c='orange')

plt.plot(training_sizes, accuracies_lda_tr, label='LDA Train', marker='o', ls='--',color='b')
plt.plot(training_sizes, accuracies_qda_tr, label='QDA Train', marker='o', ls='--', c='g')
plt.plot(training_sizes, accuracies_log_reg_tr, label='Logistic Regression Train', marker='o', ls='--',c='orange')
plt.title('Accuracy vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

## Question 9:
So how can you explain the poor performance of QDA?


```{code-cell} python
from numpy.random import normal

# Define the standard deviation of the noise
noise_level = 0.5  # Adjust this parameter based on your requirement

# Add Gaussian noise to the training and testing data
X_train_noisy = X_train + normal(0, noise_level, X_train.shape)
X_test_noisy = X_test + normal(0, noise_level, X_test.shape)

```

#### Why should we also add noise to the test data?


```{code-cell} python
# Define different training set sizes
training_sizes = [10, 100, 300, 700, 1000, len(X_train)]

# Store accuracies for each model
accuracies_lda_noisy = []
accuracies_qda_noisy = []
accuracies_log_reg_noisy = []

#Also Store TRAIN accuracies for each model
accuracies_lda_tr_noisy = []
accuracies_qda_tr_noisy = []
accuracies_log_reg_tr_noisy = []

# Evaluate models on different training sizes
for size in training_sizes:
    X_train_subset_noisy = # YOUR CODE OR ANSWER HERE
    y_train_subset = # YOUR CODE OR ANSWER HERE

    # Ensure there's at least one example of each class
    if len(np.unique(y_train_subset)) < 2:
        continue

    # LDA
    lda = # YOUR CODE OR ANSWER HERE
    acc_lda_noisy , acc_lda_tr_noisy= # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

    
    # QDA
    qda = # YOUR CODE OR ANSWER HERE
    acc_qda_noisy , acc_qda_tr_noisy = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    

    # Logistic Regression
    log_reg = # YOUR CODE OR ANSWER HERE
    acc_log_reg_noisy , acc_log_reg_tr_noisy = # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE
    # YOUR CODE OR ANSWER HERE

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(training_sizes, accuracies_lda_noisy, label='LDA', marker='o', c='b')
plt.plot(training_sizes, accuracies_qda_noisy, label='QDA', marker='o', c='g')
plt.plot(training_sizes, accuracies_log_reg_noisy, label='Logistic Regression', marker='o', c='orange')

plt.plot(training_sizes, accuracies_lda_tr_noisy, label='LDA Train', marker='o', ls='--',color='b')
plt.plot(training_sizes, accuracies_qda_tr_noisy, label='QDA Train', marker='o', ls='--', c='g')
plt.plot(training_sizes, accuracies_log_reg_tr_noisy, label='Logistic Regression Train', marker='o', ls='--',c='orange')
plt.title('Accuracy vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()
```

## Conclude

Solution:
- The numerical proble for QDA is resoved
- It outperforms both other methods

We could plot the test accuracy of QDA as a function of the noise level


```{code-cell} python
accuracies_qda_noisy_dep_level = []
noise_levels = [0, 0.1, 0.5, 1, 2, 4, 8]


#
#
# YOUR CODE OR ANSWER HERE
#
#

    
    
plt.plot(noise_levels, accuracies_qda_noisy_dep_level, label='QDA Train', marker='o', ls='--', c='g')
plt.title('QDA Accuracy vs. Level of noise on training and test Points')
plt.xlabel('Level of noise on training and test Points')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

print("Maximal test accuracy is", np.max(accuracies_qda_noisy_dep_level ))
```

Copyright: Aymeric Dieuleveut




```{code-cell} python

```