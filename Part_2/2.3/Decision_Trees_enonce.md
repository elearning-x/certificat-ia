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
  title: Decision Trees
  version: '1.0'
---

 
```{list-table} 
:header-rows: 0
:widths: 33% 34% 33%

* - ![Logo](media/logo_IPParis.png)
  - Erwan SCORNET
  - Licence CC BY-NC-ND
```

+++

# Decision Trees

## 1 - Understanding the influence of hyperparameters - Simulated data

We want to study the influence of the different hyperparameters of a decision tree on its predictive performance. 

We first use the following lines to generate and plot our dataset, containing two input variables and one binary output. We are well aware that this is nothing like a real dataset. However, it is often better to look at a method applied on synthetic data to understand its behavior.  


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate a classification dataset composed of two circles
#Generating a training set
X_train, y_train = make_circles(n_samples=200, noise=0.17)
#Generating a test set
X_test, y_test = make_circles(n_samples=10000, noise=0.17)

# Plot the generated dataset
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset')
plt.show()
```

**1) Using 'DecisionTreeClassifier', fit a decision tree with the default parameters on the training set. Compute its accuracy on the training set and test set. Comment.**


```{code-cell} python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# TO DO 

# END TO DO  
```

**2) You can plot the decision frontier with the following lines. Comment.**


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt

# Determine the range for the plot
x_min, x_max = X_train[:, 0].min() - 0.5, X_train[:, 0].max() + 0.5
y_min, y_max = X_train[:, 1].min() - 0.5, X_train[:, 1].max() + 0.5

# Create a grid of points and classify each point
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02),
                     np.arange(y_min, y_max, 0.02))
Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot the decision boundary
plt.contourf(xx, yy, Z, alpha=0.8, cmap='bwr')
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Decision Boundary of Decision Tree')
plt.show()
```

**3) We say that our algorithm overfits, as the performance on the training set is way better than that on the test set. By looking at the function 'DecisionTreeClassifier', what parameters could we change to decrease this phenomenon?**

https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

####### To Be Completed

**4) Tuning of the depth. Train a tree of depth $15$. Between this tree and the first tree you trained, which one do you prefer?**


```{code-cell} python
# TO DO 

# END TO DO 
```

**5) Now we want to find the optimal depth. To this aim, plot the test error as a function of the tree depth. Comment.**


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

#Setting the maximal depth to 15 according to the previous question
max_depth = 15
depths = np.arange(1, max_depth+1)
errors_train = []
errors_test = []

# TO DO 

# END TO DO 
```

**6) Now we want to find the optimal value for the parameter max-leaf-nodes. To this aim, plot the test error as a function of the number of leaves. Comment.** 


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

#Setting the maximal depth to 15 according to the previous question
max_leaves = 200
leaves = np.arange(2, max_leaves+1, 10)
errors_train = []
errors_test = []

# TO DO 

# END TO DO 
```

**7) Finally, we want to study the impact of pruning. To this aim, plot the test error as a function of the complexity parameter 'ccp_alpha'. Comment.**


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import zero_one_loss

#Setting the maximal depth to 15 according to the previous question
ccp = np.arange(0, 0.035, 0.001)
errors_train = []
errors_test = []

# TO DO 

# END TO DO 
```

## 2 - Applying Decision Trees on a real data set.

Now, we want to apply a decision tree to solve a real problem. Let us consider the following real estate data set.


```{code-cell} python
from sklearn.datasets import fetch_california_housing
import numpy as np
import matplotlib.pyplot as plt

house = fetch_california_housing()
X, y = house.data, house.target
feature_names = house.feature_names

feature_mapping = {
    "MedInc": "Median income in block",
    "HousAge": "Median house age in block",
    "AveRooms": "Average number of rooms",
    "AveBedrms": "Average number of bedrooms",
    "Population": "Block population",
    "AveOccup": "Average house occupancy",
    "Latitude": "House block latitude",
    "Longitude": "House block longitude",
}
```

**8) Use the command 'train_test_split' to split the data set into a training set and test set. The test set will only be used to assess the performance of our final estimator.** 


```{code-cell} python
from sklearn.model_selection import train_test_split

# TO DO 

# END TO DO 
```

**9) Train a decision tree on the above data using default parameters. Evaluate its quadratic risk on the training set and on the test set. Comment.**


```{code-cell} python
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor

# TO DO 

# END TO DO 
```

**10) Tune the complexity parameter of the pruning by cross-validation by evaluating parameter values between 0.001 and 0.03. You can make use of the command 'cross_val_score'. Comment.**


```{code-cell} python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# TO DO 

# END TO DO 
```


```{code-cell} python
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

# TO DO 

# END TO DO 
```

**11) Based on the previous question, find the pruning parameter that leads to the best predictive performance.**


```{code-cell} python
# TO DO 

# END TO DO 
```

**12) Train a tree on the whole training set with the best pruning complexity (the one determined above) and evaluate its performance on the test set.**


```{code-cell} python
from sklearn.metrics import mean_absolute_error

# TO DO 

# END TO DO 
```

**13) Plot the first level of the tree. Comment.**


```{code-cell} python
import graphviz 
from sklearn import tree


dot_data = tree.export_graphviz(clf, out_file=None, max_depth=3, feature_names=house.feature_names,  filled=True, rounded=True, special_characters=True)  
graph = graphviz.Source(dot_data)  
graph

```
