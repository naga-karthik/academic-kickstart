---
title: 'Linear Regression with PyTorch'
summary: A simple example on how to perform linear regression using PyTorch.
date: "2019-02-01T18:00:00Z"
reading_time: true  # Show estimated reading time?
share: false  # Show social sharing links?
profile: false  # Show author profile?
comments: true  # Show comments?
tags:
  - Machine Learning
  - PyTorch
  - Linear Regression
---

This is just a simple introduction to coding in PyTorch. We’ll build a two-dimensional linear regression model and see how the model performs by observing the graphs at the output. So, let’s begin!

There are a million ways to create a linear regression model if we have the data readily available (or downloaded from somewhere). In that case, we will not be able to take advantage of PyTorch’s Dataset class which allows us to create datasets of our own and test them (interesting, right?). Therefore, in this article, I will be showing how to create an artificial dataset using the Dataset class and run our model. PyTorch also has this cool feature of letting users build their own custom modules of any kind, be it linear regression model, logistic regression model, neural network model etc. However, it goes without saying that there are in-built functions also for the same purposes. Enough of the jargon now, let’s dive into the code!

First off, we will begin by importing all the necessary packages to build our model.

```python
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
torch.manual_seed(1)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
```

_Dataset_ is the class that contains the artificial dataset (in the form of a subclass within the main class) that we 
will define and _DataLoader_ is responsible for loading the dataset. For creating our custom modules, the nn module must 
be imported and for choosing the optimizer of our choice, the optim module must be used.

```python
# Creating the artificial dataset
class Data2d(Dataset):
    
    # Constructor
    def __init__(self):
        self.x = torch.zeros(20,2)
        self.x[:,0] = torch.arange(-1,1,0.1)
        self.x[:,1] = torch.arange(-1,1,0.1)
        
        self.w = torch.tensor([[1.0], [1.0]])
        self.b = 1
        self.f = torch.mm(self.x, self.w) + self.b
        
        self.y = self.f + 0.1*torch.randn((self.x.shape[0],1))
        self.len = self.x.shape[0]
    
    # Getter
    def __getitem__(self, index):
        return self.x[index], self.y[index]
    
    # Getting the length
    def __len__(self):
        return self.len
# Instantiation of the class  
my_data = Data2d()

```
We have randomly chosen any set of values of **x** , which I have assumed to a 2D matrix (for no particular reason, 
it can be anything!). The slope and bias **w** and **b** respectively are defined. The equation of a line is written 
and stored in the variable **f** . I have added some random Gaussian noise to the equation with a standard deviation of 
0.1 so that it isn’t just a straight line.

Next up in the code is a function we have defined that plots the 2D plane for better visualization purposes.

```python
# Defining a function for plotting the plane
def plane2D(model,dataset,n=0):
      
    w1 = model.state_dict()['linear.weight'].numpy()[0][0]
    w2 = model.state_dict()['linear.weight'].numpy()[0][1]
    b = model.state_dict()['linear.bias'].numpy()
    
    x1 = dataset.x[:,0].view(-1,1).numpy()
    x2 = dataset.x[:,1].view(-1,1).numpy()
    y = dataset.y.numpy()
    X,Y = np.meshgrid(np.arange(x1.min(),x1.max(),0.05),np.arange(x2.min(), x2.max(), 0.05))
    
    yhat = w1*X + w2*Y + b
    
    # plotting
    fig = plt.figure()
    ax = fig.gca(projection = '3d')
    
    ax.plot(x1[:,0], x2[:,0], y[:,0], 'ro', label = 'y')
    ax.plot_surface(X,Y,yhat)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    ax.legend()
    plt.title('Estimated plane iteration: '+ str(n))
    plt.show()

```
PyTorch defines the parameters, that is, the slope and the bias randomly. We are storing those values by calling the 
**model.state_dict()** function and explicitly mentioning them by **linear.weight** and **linear.bias**. We will be 
passing the artificial dataset as an argument to the plane2D function, from where the data is stored in the corresponding 
variables. The remaining part of the code is just the standard way of plotting 3D data.

Note: The **.numpy()** function is used wherever there is a need of converting the data from a tensor (PyTorch processes 
data in the form of tensors) to a numpy array.

```python
# Creating a linear regression model
class lin_reg(nn.Module):

    def __init__(self, in_feat, out_feat):
        super(lin_reg, self).__init__()
        self.linear = nn.Linear(in_feat, out_feat)
        
    def forward(self,x):
        yhat = self.linear(x)
        return yhat
# Instantiation of an object
model = lin_reg(2,1)
print("The parameters: ", model.state_dict())

```
**nn.Module** is the base class for all modules in PyTorch. All the models that we define must be subclasses within this 
class. The **super()** function returns a proxy object that delegates the methods calls to a class. The in-built function
**nn.Linear** is used which just takes the number of input and output features as arguments and simply applies the linear 
transformation. It can be looked at in this way, the **__init__()** function is where the model variables are initialized 
and the **forward()** function is where these variables are used to predict the output.

Now, all that’s left is to define the loss function, the optimizer and then train our model!

```python
# Parameters
criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr = 0.01) 
# Training data object which loads the artificial data
trainloader = DataLoader(dataset = my_data, batch_size = 2)
# Training the model
Loss = []  # variable for storing losses after each epoch
epochs = 100
print('Before training:')
plane2D(model, my_data)
def train_model(epochs):
    for epoch in range(epochs):
        for x,y in trainloader:
            yhat = model(x)
            loss = criterion(yhat,y)
            Loss.append(loss.item())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
# Calling the training function          
train_model(epochs)
print("After training: ")
plane2D(model, my_data, epochs)

```
I have decided to use the mean-squared error as the loss function. The in-built function is used here, however, a 
function can be created which does the same.

```python
def criterion(yhat,y):
    out = torch.mean((yhat - y)**2)
    return out
```

The optimizer used is Stochastic Gradient Descent (SGD). The model parameters must be passed as the argument and an 
appropriate learning rate must be defined. The way in which PyTorch calculates the error and updates the weights is that, 
firstly, all the gradients are set to zero using **.zero_grad()** function, then the derivatives are calculated using the 
**.backward()** function. Finally, the weights are updated using the **.step()** function. The loss after the completion of each 
epoch is stored which helps in plotting (after the training is completed). It is just to see how the model performed 
during the training.

And that’s it! I hope that this introductory tutorial in PyTorch helps. Please feel free to comment/suggest anything 
related to the above article!
