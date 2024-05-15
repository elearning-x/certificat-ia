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
  title: Neural networks correction
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


    
![png](media/Part_2/2.8/Neural_networks_correction_3_0.png)
    



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




    (torch.Tensor, torch.Tensor)



**1) Split the training set into a new training set and a validation set (80%/20%). The validation set will be used to train the network, whereas the test set will only be used in the end to evaluate the performance of our network.** 


```{code-cell} python
from sklearn.model_selection import train_test_split

x_train, x_valid, y_train, y_valid = train_test_split(x_data, y_data, test_size=0.2, shuffle=True)
```

**2) What is the size of all the objects you have created at the previous question?**


```{code-cell} python
# You can print the shape of an object with the .shape command

print('x_train:', x_train.shape)
print('y_train:', y_train.shape)
print('x_valid:', x_valid.shape)
print('y_valid:', y_valid.shape)

#This was expected: the inputs are of size two, the output is of size one. 
# We have 80% / 20% of the initial data divided into x_train/x_valid and y_train/y_valid
```

    x_train: torch.Size([8000, 2])
    y_train: torch.Size([8000])
    x_valid: torch.Size([2000, 2])
    y_valid: torch.Size([2000])


**3) By looking at the documentation below, define a 'NeuralNetwork' class for a network architecture composed of two hidden layers (64 neurons per layer, sigmoid activation function) and an output layer.**

 https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html 


```{code-cell} python
input_dim = 2
hidden_dim = 64
output_dim = 1


class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Define the layers
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid2 = nn.Sigmoid()

    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid2(out)
        return out
  
```

**4) You have just created the class but no networks have been created. Instantiate the class by creating one network that belongs to it.** 


```{code-cell} python
model = NeuralNetwork()
print(model)
```

    NeuralNetwork(
      (fc1): Linear(in_features=2, out_features=64, bias=True)
      (sigmoid): Sigmoid()
      (fc2): Linear(in_features=64, out_features=1, bias=True)
      (sigmoid2): Sigmoid()
    )


**5) Compute the prediction of the network for the first ten observations in the test set. Beware, since you manipulate tensors, operations (even the most elementary ones) must be done through 'torch' environment and not 'numpy' environment.** 


```{code-cell} python
result = model(x_train[0:10,:])
result = torch.reshape(result, (-1,))

print("The estimated probabilities output by the network for the first ten observations are", result)

print("The corresponding predictions are", 1*(result>1/2))

print("The corresponding lavels are", y_train[0:10])

torch.eq(y_train[0:10], 1*(result>1/2))


print("The resulting accuracy for the first ten observations is " , torch.sum(torch.eq(y_train[0:10], 1*(result>1/2)))/10 )
```

    The estimated probabilities output by the network for the first ten observations are tensor([0.5139, 0.5214, 0.5180, 0.5162, 0.5160, 0.5145, 0.5163, 0.5226, 0.5231,
            0.5183], grad_fn=<ReshapeAliasBackward0>)
    The corresponding predictions are tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    The corresponding lavels are tensor([0., 1., 1., 0., 1., 0., 1., 1., 0., 1.])
    The resulting accuracy for the first ten observations is  tensor(0.6000)


**6) Discuss the error of the network.**


```{code-cell} python
# The error of the network is quite high, which is quite logical since the network has not seen any data. 
# Indeed, in the code above, the class NeuralNetwork does not contain any observations from the dataset.
# Besides, we have not specified the training procedure of the network. 
# In conclusion, the network has not been trained. 
# The prediction we obtain here results from the random weight initialization and is thus inaccurate. 
```

## 2 - Train the network architecture

In order to train the network, we need to define a loss and an optimizer. This is done in the following code. 


```{code-cell} python
criterion = torch.nn.CrossEntropyLoss()
# Optimizers specified in the torch.optim package
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
```

**7) Why do we use the cross-entropy loss? Why do we need to specify model.parameters?**

**Answer** The cross-entropy loss is the most adapted loss to a classification problem. We need to specify the set of trainable parameters that will be optimized via the optimization procedure. With the above command, we impose that all parameters of the network will be trained during the optimization steps. 

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

    Epoch:    0 | Train Loss: 1.73153, Accuracy: 50.04% | Validation Loss: 3.81191, Accuracy: 49.85%
    Epoch:    1 | Train Loss: 1.73151, Accuracy: 49.35% | Validation Loss: 3.81184, Accuracy: 49.70%
    Epoch:    2 | Train Loss: 1.73147, Accuracy: 47.56% | Validation Loss: 3.81180, Accuracy: 48.20%
    Epoch:    3 | Train Loss: 1.73142, Accuracy: 51.86% | Validation Loss: 3.81176, Accuracy: 51.30%
    Epoch:    4 | Train Loss: 1.73140, Accuracy: 54.60% | Validation Loss: 3.81172, Accuracy: 52.75%
    Epoch:    5 | Train Loss: 1.73136, Accuracy: 51.16% | Validation Loss: 3.81169, Accuracy: 50.15%
    Epoch:    6 | Train Loss: 1.73133, Accuracy: 49.96% | Validation Loss: 3.81163, Accuracy: 50.15%
    Epoch:    7 | Train Loss: 1.73127, Accuracy: 49.96% | Validation Loss: 3.81162, Accuracy: 50.15%
    Epoch:    8 | Train Loss: 1.73122, Accuracy: 49.96% | Validation Loss: 3.81154, Accuracy: 50.15%
    Epoch:    9 | Train Loss: 1.73117, Accuracy: 49.96% | Validation Loss: 3.81153, Accuracy: 50.15%
    Epoch:   10 | Train Loss: 1.73114, Accuracy: 49.96% | Validation Loss: 3.81146, Accuracy: 50.15%
    Epoch:   11 | Train Loss: 1.73109, Accuracy: 49.96% | Validation Loss: 3.81142, Accuracy: 50.15%
    Epoch:   12 | Train Loss: 1.73104, Accuracy: 49.96% | Validation Loss: 3.81136, Accuracy: 50.15%
    Epoch:   13 | Train Loss: 1.73098, Accuracy: 49.96% | Validation Loss: 3.81129, Accuracy: 50.15%
    Epoch:   14 | Train Loss: 1.73094, Accuracy: 49.96% | Validation Loss: 3.81123, Accuracy: 50.15%
    Epoch:   15 | Train Loss: 1.73087, Accuracy: 49.96% | Validation Loss: 3.81116, Accuracy: 50.15%
    Epoch:   16 | Train Loss: 1.73080, Accuracy: 50.01% | Validation Loss: 3.81110, Accuracy: 50.20%
    Epoch:   17 | Train Loss: 1.73073, Accuracy: 50.30% | Validation Loss: 3.81102, Accuracy: 50.40%
    Epoch:   18 | Train Loss: 1.73064, Accuracy: 50.54% | Validation Loss: 3.81097, Accuracy: 50.70%
    Epoch:   19 | Train Loss: 1.73058, Accuracy: 51.11% | Validation Loss: 3.81088, Accuracy: 51.15%
    Epoch:   20 | Train Loss: 1.73049, Accuracy: 52.10% | Validation Loss: 3.81077, Accuracy: 52.45%
    Epoch:   21 | Train Loss: 1.73041, Accuracy: 53.38% | Validation Loss: 3.81064, Accuracy: 54.90%
    Epoch:   22 | Train Loss: 1.73029, Accuracy: 54.17% | Validation Loss: 3.81054, Accuracy: 55.85%
    Epoch:   23 | Train Loss: 1.73017, Accuracy: 57.40% | Validation Loss: 3.81041, Accuracy: 55.50%
    Epoch:   24 | Train Loss: 1.73005, Accuracy: 57.86% | Validation Loss: 3.81029, Accuracy: 53.70%
    Epoch:   25 | Train Loss: 1.72988, Accuracy: 56.48% | Validation Loss: 3.81020, Accuracy: 55.45%
    Epoch:   26 | Train Loss: 1.72973, Accuracy: 58.01% | Validation Loss: 3.81002, Accuracy: 57.60%
    Epoch:   27 | Train Loss: 1.72958, Accuracy: 58.76% | Validation Loss: 3.80982, Accuracy: 59.75%
    Epoch:   28 | Train Loss: 1.72938, Accuracy: 62.15% | Validation Loss: 3.80959, Accuracy: 59.60%
    Epoch:   29 | Train Loss: 1.72914, Accuracy: 61.09% | Validation Loss: 3.80937, Accuracy: 60.45%
    Epoch:   30 | Train Loss: 1.72893, Accuracy: 61.29% | Validation Loss: 3.80908, Accuracy: 63.00%
    Epoch:   31 | Train Loss: 1.72858, Accuracy: 64.15% | Validation Loss: 3.80878, Accuracy: 61.65%
    Epoch:   32 | Train Loss: 1.72831, Accuracy: 63.23% | Validation Loss: 3.80841, Accuracy: 64.65%
    Epoch:   33 | Train Loss: 1.72796, Accuracy: 64.51% | Validation Loss: 3.80795, Accuracy: 67.55%
    Epoch:   34 | Train Loss: 1.72758, Accuracy: 64.66% | Validation Loss: 3.80751, Accuracy: 68.70%
    Epoch:   35 | Train Loss: 1.72710, Accuracy: 65.72% | Validation Loss: 3.80707, Accuracy: 67.45%
    Epoch:   36 | Train Loss: 1.72663, Accuracy: 65.01% | Validation Loss: 3.80657, Accuracy: 66.70%
    Epoch:   37 | Train Loss: 1.72602, Accuracy: 66.25% | Validation Loss: 3.80602, Accuracy: 66.55%
    Epoch:   38 | Train Loss: 1.72542, Accuracy: 65.99% | Validation Loss: 3.80527, Accuracy: 68.30%
    Epoch:   39 | Train Loss: 1.72475, Accuracy: 67.65% | Validation Loss: 3.80461, Accuracy: 66.50%
    Epoch:   40 | Train Loss: 1.72396, Accuracy: 66.60% | Validation Loss: 3.80367, Accuracy: 69.85%
    Epoch:   41 | Train Loss: 1.72310, Accuracy: 67.66% | Validation Loss: 3.80275, Accuracy: 68.55%
    Epoch:   42 | Train Loss: 1.72215, Accuracy: 67.76% | Validation Loss: 3.80185, Accuracy: 68.30%
    Epoch:   43 | Train Loss: 1.72115, Accuracy: 67.38% | Validation Loss: 3.80058, Accuracy: 69.40%
    Epoch:   44 | Train Loss: 1.72001, Accuracy: 68.64% | Validation Loss: 3.79959, Accuracy: 69.45%
    Epoch:   45 | Train Loss: 1.71876, Accuracy: 68.64% | Validation Loss: 3.79819, Accuracy: 69.85%
    Epoch:   46 | Train Loss: 1.71754, Accuracy: 68.99% | Validation Loss: 3.79681, Accuracy: 70.15%
    Epoch:   47 | Train Loss: 1.71618, Accuracy: 69.79% | Validation Loss: 3.79548, Accuracy: 70.25%
    Epoch:   48 | Train Loss: 1.71484, Accuracy: 69.85% | Validation Loss: 3.79399, Accuracy: 70.90%
    Epoch:   49 | Train Loss: 1.71329, Accuracy: 70.19% | Validation Loss: 3.79230, Accuracy: 70.85%
    Epoch:   50 | Train Loss: 1.71159, Accuracy: 70.61% | Validation Loss: 3.79080, Accuracy: 71.50%
    Epoch:   51 | Train Loss: 1.71025, Accuracy: 70.88% | Validation Loss: 3.78898, Accuracy: 72.00%
    Epoch:   52 | Train Loss: 1.70851, Accuracy: 71.40% | Validation Loss: 3.78726, Accuracy: 72.15%
    Epoch:   53 | Train Loss: 1.70681, Accuracy: 71.36% | Validation Loss: 3.78561, Accuracy: 72.15%
    Epoch:   54 | Train Loss: 1.70523, Accuracy: 71.76% | Validation Loss: 3.78404, Accuracy: 72.75%
    Epoch:   55 | Train Loss: 1.70350, Accuracy: 71.54% | Validation Loss: 3.78249, Accuracy: 72.50%
    Epoch:   56 | Train Loss: 1.70206, Accuracy: 71.86% | Validation Loss: 3.78077, Accuracy: 72.40%
    Epoch:   57 | Train Loss: 1.70048, Accuracy: 71.81% | Validation Loss: 3.77943, Accuracy: 72.45%
    Epoch:   58 | Train Loss: 1.69904, Accuracy: 71.75% | Validation Loss: 3.77778, Accuracy: 72.50%
    Epoch:   59 | Train Loss: 1.69759, Accuracy: 71.86% | Validation Loss: 3.77655, Accuracy: 72.30%
    Epoch:   60 | Train Loss: 1.69658, Accuracy: 71.69% | Validation Loss: 3.77533, Accuracy: 72.15%
    Epoch:   61 | Train Loss: 1.69493, Accuracy: 71.70% | Validation Loss: 3.77411, Accuracy: 72.20%
    Epoch:   62 | Train Loss: 1.69410, Accuracy: 71.97% | Validation Loss: 3.77291, Accuracy: 72.20%
    Epoch:   63 | Train Loss: 1.69317, Accuracy: 71.49% | Validation Loss: 3.77196, Accuracy: 72.10%
    Epoch:   64 | Train Loss: 1.69226, Accuracy: 71.67% | Validation Loss: 3.77082, Accuracy: 72.30%
    Epoch:   65 | Train Loss: 1.69093, Accuracy: 71.85% | Validation Loss: 3.76988, Accuracy: 72.50%
    Epoch:   66 | Train Loss: 1.69047, Accuracy: 71.78% | Validation Loss: 3.76917, Accuracy: 72.25%
    Epoch:   67 | Train Loss: 1.68957, Accuracy: 71.59% | Validation Loss: 3.76833, Accuracy: 72.30%
    Epoch:   68 | Train Loss: 1.68842, Accuracy: 71.84% | Validation Loss: 3.76738, Accuracy: 72.50%
    Epoch:   69 | Train Loss: 1.68789, Accuracy: 71.95% | Validation Loss: 3.76676, Accuracy: 72.50%
    Epoch:   70 | Train Loss: 1.68734, Accuracy: 71.94% | Validation Loss: 3.76603, Accuracy: 72.60%
    Epoch:   71 | Train Loss: 1.68698, Accuracy: 71.71% | Validation Loss: 3.76543, Accuracy: 72.50%
    Epoch:   72 | Train Loss: 1.68600, Accuracy: 71.59% | Validation Loss: 3.76478, Accuracy: 72.80%
    Epoch:   73 | Train Loss: 1.68515, Accuracy: 71.74% | Validation Loss: 3.76454, Accuracy: 72.65%
    Epoch:   74 | Train Loss: 1.68508, Accuracy: 71.51% | Validation Loss: 3.76379, Accuracy: 72.55%
    Epoch:   75 | Train Loss: 1.68466, Accuracy: 71.67% | Validation Loss: 3.76328, Accuracy: 72.70%
    Epoch:   76 | Train Loss: 1.68395, Accuracy: 71.71% | Validation Loss: 3.76277, Accuracy: 72.95%
    Epoch:   77 | Train Loss: 1.68382, Accuracy: 71.61% | Validation Loss: 3.76248, Accuracy: 72.25%
    Epoch:   78 | Train Loss: 1.68388, Accuracy: 71.64% | Validation Loss: 3.76201, Accuracy: 72.60%
    Epoch:   79 | Train Loss: 1.68288, Accuracy: 71.59% | Validation Loss: 3.76148, Accuracy: 72.50%
    Epoch:   80 | Train Loss: 1.68280, Accuracy: 71.89% | Validation Loss: 3.76118, Accuracy: 72.55%
    Epoch:   81 | Train Loss: 1.68236, Accuracy: 71.56% | Validation Loss: 3.76080, Accuracy: 72.50%
    Epoch:   82 | Train Loss: 1.68190, Accuracy: 71.79% | Validation Loss: 3.76042, Accuracy: 72.70%
    Epoch:   83 | Train Loss: 1.68157, Accuracy: 71.56% | Validation Loss: 3.76009, Accuracy: 72.75%
    Epoch:   84 | Train Loss: 1.68152, Accuracy: 71.76% | Validation Loss: 3.75992, Accuracy: 72.75%
    Epoch:   85 | Train Loss: 1.68133, Accuracy: 71.84% | Validation Loss: 3.75958, Accuracy: 72.95%
    Epoch:   86 | Train Loss: 1.68097, Accuracy: 71.62% | Validation Loss: 3.75932, Accuracy: 72.35%
    Epoch:   87 | Train Loss: 1.68051, Accuracy: 71.58% | Validation Loss: 3.75903, Accuracy: 72.70%
    Epoch:   88 | Train Loss: 1.68072, Accuracy: 71.75% | Validation Loss: 3.75884, Accuracy: 72.80%
    Epoch:   89 | Train Loss: 1.68034, Accuracy: 71.80% | Validation Loss: 3.75884, Accuracy: 72.75%
    Epoch:   90 | Train Loss: 1.68089, Accuracy: 71.65% | Validation Loss: 3.75850, Accuracy: 72.80%
    Epoch:   91 | Train Loss: 1.67978, Accuracy: 71.69% | Validation Loss: 3.75819, Accuracy: 72.65%
    Epoch:   92 | Train Loss: 1.67995, Accuracy: 71.69% | Validation Loss: 3.75804, Accuracy: 72.65%
    Epoch:   93 | Train Loss: 1.67894, Accuracy: 71.72% | Validation Loss: 3.75849, Accuracy: 72.50%
    Epoch:   94 | Train Loss: 1.67907, Accuracy: 71.70% | Validation Loss: 3.75749, Accuracy: 72.95%
    Epoch:   95 | Train Loss: 1.67879, Accuracy: 71.79% | Validation Loss: 3.75752, Accuracy: 72.35%
    Epoch:   96 | Train Loss: 1.67889, Accuracy: 71.85% | Validation Loss: 3.75729, Accuracy: 72.95%
    Epoch:   97 | Train Loss: 1.67920, Accuracy: 71.62% | Validation Loss: 3.75719, Accuracy: 72.80%
    Epoch:   98 | Train Loss: 1.67875, Accuracy: 71.81% | Validation Loss: 3.75717, Accuracy: 72.80%
    Epoch:   99 | Train Loss: 1.67887, Accuracy: 71.71% | Validation Loss: 3.75674, Accuracy: 72.85%


**9) Plot the evolution of the training and validation accuracy as a function of the number of epochs. Comment.**


```{code-cell} python
plt.plot(np.arange(num_epochs), train_acc_values, label='Training accuracy')
plt.plot(np.arange(num_epochs), valid_acc_values, label='Validation accuracy')
plt.title('Training & Validation Accuracy Curves')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend()
plt.show()

# The accuracy curves seem to reach a plateau, 
# which indicates that increasing the number of epochs would not be beneficial to the model. 
```


    
![png](media/Part_2/2.8/Neural_networks_correction_25_0.png)
    


**10) Evaluate the performance of the final neural network on the test set.**


```{code-cell} python
predictions = model(x_data_test)  # Predictions for test data
predictions = predictions.squeeze()
acc = accuracy_fn(y_data_test>1/2, predictions>1/2)

print("The accuracy of the final neural network is:", acc)

#This seems to be of the same magnitude as the error obtained on the validation set during the training. 
```

    The accuracy of the final neural network is: 72.0

