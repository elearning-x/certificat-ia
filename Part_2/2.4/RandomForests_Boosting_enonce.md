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
  title: RandomForests Boosting
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

# Random Forests and Tree Boosting



## 1 - Regression 

In this part, we apply Random Forests and Tree Boosting on the real estate dataset that we have seen before. 

**1) Load the data, split the data into a train set (80%) and a test set (20%).**


```{code-cell} python
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split

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

# TO DO 

# END TO DO 
```

**2) Train a random forests with default parameters on these data and evaluate its performance on the training set and test set. Comment.** 


```{code-cell} python
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# TO DO 

# END TO DO 
```

**3) Based on the lecture, which parameter could have the biggest impact on the predictive performances of the forest? Plot the error of the forest as a function of this parameter.** 


```{code-cell} python
# The parameter that can have the greatest impact on the prediction is the max-features 
# (the number of randomly selected directions for splitting in each node).

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score

# TO DO 

# END TO DO 
```

**4) Train a forest with the best parameter found above. Comment.**


```{code-cell} python
# TO DO 

# END TO DO 
```

**5) Now, optimize jointly the parameters 'max-features' and 'max-depth' of the random forest. You can use the function 'GridSearchCV'.**


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# TO DO 

# END TO DO 
```

**6) Train the best resulting model and compute its error.** 


```{code-cell} python
# TO DO 

# END TO DO 
```

**7) Train a Gradient Boosting Decision Tree using 'GradientBoostingRegressor'. Tune the hyperparameters 'learning_rate' and 'max_depth' via cross-validation as above. Comment the results.**


```{code-cell} python
from sklearn.ensemble import GradientBoostingRegressor

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, mean_squared_error

# TO DO 

# END TO DO 
```


```{code-cell} python
# TO DO 

# END TO DO 
```

## 2 - Classification on a Toy dataset


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# Create an imbalanced dataset
X, y = make_classification(n_samples=5000,weights=[0.02, 0.98],
                           random_state=0,n_clusters_per_class=1)

# Generate a classification dataset composed of two circles
#X, y = make_circles(n_samples=(200, 10), noise=0.1)

# Plot the generated dataset
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset')
plt.show()
```

**8) Split the data set into a train and a test set. Train a random forest with max_depth = 2 on the training set. Plot its performance with a confusion matrix on the test set. Comment.**


```{code-cell} python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle = True)


# TO DO 




# END TO DO 
```

In presence of an imbalanced data set, we need to pay attention to two different things: 
- when dividing the dataset into several parts (training/test), we want to keep the same proportion of observations in the resulting datasets. To do so, we use stratification. 
- we need to rebalance the training set so that it contains roughly the same proportions for the two classes. This can be done via the parameter 'class_weight'. It thus helps the training process by weighting up observations from the minority class. 
No matter what you do on the training set, you are not allowed to change the test set. It must reflect the true distribution of data. 

**9) Implement these modifications, train a forest and compare its performances.**


```{code-cell} python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# TO DO 

# END TO DO 
```
