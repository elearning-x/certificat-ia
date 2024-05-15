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
  title: Neural networks
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

# Neural Network - How to train a neural network on tabular data?


## 1 - Create the network architecture

Let us consider the toy dataset below. 


```{code-cell} python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles

# Generate a classification dataset composed of two circles
X_train, y_train = make_circles(n_samples=10000, noise=0.17)
X_test, y_test = make_circles(n_samples=10000, noise=0.17)

# Plot the generated dataset
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap='bwr')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Generated Dataset')
plt.show()
```


```{code-cell} python
import pandas as pd
import torch
from torch import nn

# Create a DataFrame from the array
df_train = pd.DataFrame(X_train)
df_test = pd.DataFrame(X_test)

#Create training data
x_data = torch.tensor(df_train.values, dtype=torch.float32)
y_data = torch.tensor(y_train, dtype=torch.float32)

#Create test data
x_data_test = torch.tensor(df_test.values, dtype=torch.float32)
y_data_test = torch.tensor(y_test, dtype=torch.float32)

type(x_data), type(y_data)
```

**1) Split the training set into a new training set and a validation set (80%/20%). The validation set will be used to train the network, whereas the test set will only be used in the end to evaluate the performance of our network.** 


```{code-cell} python
from sklearn.model_selection import train_test_split

# TO DO 

# END TO DO 
```

**2) What is the size of all the objects you have created at the previous question?**


```{code-cell} python
# TO DO 

# END TO DO 
```

**3) By looking at the documentation below, define a 'NeuralNetwork' class for a network architecture composed of two hidden layers (64 neurons per layer, sigmoid activation function) and an output layer.**

 https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html 


```{code-cell} python
input_dim = 2
hidden_dim = 64
output_dim = 1


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # TO DO 

        # END TO DO 

    def forward(self, x):
        # TO DO 

        # END TO DO 
        return out
  
```

**4) You have just created the class but no networks have been created. Instantiate the class by creating one network that belongs to it.** 


```{code-cell} python
# TO DO 

# END TO DO 
```

**5) Compute the prediction of the network for the first ten observations in the test set. Beware, since you manipulate tensors, operations (even the most elementary ones) must be done through 'torch' environment and not 'numpy' environment.** 


```{code-cell} python
# TO DO 

# END TO DO 
```

**6) Discuss the error of the network.**


```{code-cell} python

```

## 2 - Train the network architecture

In order to train the network, we need to define a loss and an optimizer. This is done in the following code. 


```{code-cell} python
criterion = torch.nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

**7) Why do we use the cross-entropy loss? Why do we need to specify model.parameters?**



**8) Use the following code to train the network. Make sure to understand each line.**


```{code-cell} python
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = correct / len(y_pred) * 100
    return acc

#Initialize the model 

model = NeuralNetwork()

# Hyperparameters
num_epochs = 100
batch_size = 32

# Create TensorDataset and DataLoader for training data
train_dataset = TensorDataset(x_train, y_train)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Define loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

#Store the values at each epoch in the following objects
epoch_count, train_loss_values, valid_loss_values, train_acc_values, valid_acc_values = [], [], [], [], []

# Training loop
for epoch in range(num_epochs):
    train_loss_values_temp = []
    train_acc_values_temp = []

    for inputs, targets in train_dataloader:
        # Forward pass - compute the predictions
        outputs = model(inputs)
        outputs = outputs.squeeze()
        
        #Compute the loss used for the optimization
        loss = criterion(outputs, targets)
        
        #Compute the accuracy, used for monitoring the model through epochs.
        acc = accuracy_fn(targets>1/2, outputs>1/2)
        
        train_loss_values_temp.append(loss.detach().numpy()/len(targets))
        train_acc_values_temp.append(acc)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Print the loss after every epoch
    #print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

    # Put the model in evaluation mode
    model.eval() 

    with torch.inference_mode():
        
        #Compute the predictions on the validation set
        output_valid = model(x_valid)   
        output_valid = output_valid.squeeze()
        
        # Compute the loss and the accuracy on the validation set
        valid_loss = criterion(output_valid, y_valid)
        valid_acc = accuracy_fn(y_valid>1/2, output_valid>1/2)    

    # Print the loss and the accuracy on the training set and validation set for each epoch
    
    print(f'Epoch: {epoch:4.0f} | Train Loss: {np.mean(train_loss_values_temp):.5f}, Accuracy: {np.mean(train_acc_values_temp):.2f}% | Validation Loss: {valid_loss/len(y_valid):.5f}, Accuracy: {valid_acc:.2f}%')
    valid_acc_values.append(valid_acc)
    valid_loss_values.append(valid_loss.detach().numpy()/ len(y_valid))
    train_acc_values.append(np.mean(train_acc_values_temp))
    train_loss_values.append(np.mean(train_loss_values_temp))


```

**9) Plot the evolution of the training and validation accuracy as a function of the number of epochs. Comment.**


```{code-cell} python
# TO DO 

# END TO DO 
```

**10) Evaluate the performance of the final neural network on the test set.**


```{code-cell} python
# TO DO 

# END TO DO 
```
