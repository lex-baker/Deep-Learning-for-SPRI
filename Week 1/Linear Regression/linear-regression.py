#!/usr/bin/env python
# coding: utf-8

# ## Gradient Descent and Linear Regression with PyTorch
# 
# ### Part 2 of "Deep Learning with Pytorch: Zero to GANs"
# 
# This tutorial series is a hands-on beginner-friendly introduction to deep learning using [PyTorch](https://pytorch.org), an open-source neural networks library. These tutorials take a practical and coding-focused approach. The best way to learn the material is to execute the code and experiment with it yourself. Check out the full series here:
# 
# 1. [PyTorch Basics: Tensors & Gradients](https://jovian.ai/aakashns/01-pytorch-basics)
# 2. [Gradient Descent & Linear Regression](https://jovian.ai/aakashns/02-linear-regression)
# 3. [Working with Images & Logistic Regression](https://jovian.ai/aakashns/03-logistic-regression) 
# 4. [Training Deep Neural Networks on a GPU](https://jovian.ai/aakashns/04-feedforward-nn)
# 5. [Image Classification using Convolutional Neural Networks](https://jovian.ai/aakashns/05-cifar10-cnn)
# 6. [Data Augmentation, Regularization and ResNets](https://jovian.ai/aakashns/05b-cifar10-resnet)
# 7. [Generating Images using Generative Adversarial Networks](https://jovian.ai/aakashns/06b-anime-dcgan/)
# 

# This tutorial covers the following topics:
# 
# - Introduction to linear regression and gradient descent
# - Implementing a linear regression model using PyTorch tensors
# - Training a linear regression model using the gradient descent algorithm
# - Implementing gradient descent and linear regression using PyTorch built-in

# ### How to run the code
# 
# This tutorial is an executable [Jupyter notebook](https://jupyter.org) hosted on [Jovian](https://www.jovian.ai) (don't worry if these terms seem unfamiliar; we'll learn more about them soon). You can _run_ this tutorial and experiment with the code examples in a couple of ways: *using free online resources* (recommended) or *on your computer*.
# 
# #### Option 1: Running using free online resources (1-click, recommended)
# 
# The easiest way to start executing the code is to click the **Run** button at the top of this page and select **Run on Colab**. [Google Colab](https://colab.research.google.com) is a free online platform for running Jupyter notebooks using Google's cloud infrastructure. You can also select "Run on Binder" or "Run on Kaggle" if you face issues running the notebook on Google Colab.
# 
# 
# #### Option 2: Running on your computer locally
# 
# To run the code on your computer locally, you'll need to set up [Python](https://www.python.org), download the notebook and install the required libraries. We recommend using the [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) distribution of Python. Click the **Run** button at the top of this page, select the **Run Locally** option, and follow the instructions.
# 
# >  **Jupyter Notebooks**: This tutorial is a [Jupyter notebook](https://jupyter.org) - a document made of _cells_. Each cell can contain code written in Python or explanations in plain English. You can execute code cells and view the results, e.g., numbers, messages, graphs, tables, files, etc. instantly within the notebook. Jupyter is a powerful platform for experimentation and analysis. Don't be afraid to mess around with the code & break things - you'll learn a lot by encountering and fixing errors. You can use the "Kernel > Restart & Clear Output" or "Edit > Clear Outputs" menu option to clear all outputs and start again from the top.

# Before we begin, we need to install the required libraries. The installation of PyTorch may differ based on your operating system / cloud environment. You can find detailed installation instructions here: https://pytorch.org .

# In[1]:


# Uncomment and run the appropriate command for your operating system, if required

# Linux / Binder
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# Windows
# !pip install numpy torch==1.7.0+cpu torchvision==0.8.1+cpu torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html

# MacOS
# !pip install numpy torch torchvision torchaudio


# ## Introduction to Linear Regression
# 
# In this tutorial, we'll discuss one of the foundational algorithms in machine learning: *Linear regression*. We'll create a model that predicts crop yields for apples and oranges (*target variables*) by looking at the average temperature, rainfall, and humidity (*input variables or features*) in a region. Here's the training data:
# 
# ![linear-regression-training-data](https://i.imgur.com/6Ujttb4.png)
# 
# In a linear regression model, each target variable is estimated to be a weighted sum of the input variables, offset by some constant, known as a bias :
# 
# ```
# yield_apple  = w11 * temp + w12 * rainfall + w13 * humidity + b1
# yield_orange = w21 * temp + w22 * rainfall + w23 * humidity + b2
# ```
# 
# Visually, it means that the yield of apples is a linear or planar function of temperature, rainfall and humidity:
# 
# ![linear-regression-graph](https://i.imgur.com/4DJ9f8X.png)

# The *learning* part of linear regression is to figure out a set of weights `w11, w12,... w23, b1 & b2` using the training data, to make accurate predictions for new data. The _learned_ weights will be used to predict the yields for apples and oranges in a new region using the average temperature, rainfall, and humidity for that region. 
# 
# We'll _train_ our model by adjusting the weights slightly many times to make better predictions, using an optimization technique called *gradient descent*. Let's begin by importing Numpy and PyTorch.

# In[2]:


import numpy as np
import torch


# ## Training data
# 
# We can represent the training data using two matrices: `inputs` and `targets`, each with one row per observation, and one column per variable.

# In[3]:


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70]], dtype='float32')


# In[4]:


# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119]], dtype='float32')


# We've separated the input and target variables because we'll operate on them separately. Also, we've created numpy arrays, because this is typically how you would work with training data: read some CSV files as numpy arrays, do some processing, and then convert them to PyTorch tensors.
# 
# Let's convert the arrays to PyTorch tensors.

# In[5]:


# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print(inputs)
print(targets)


# ## Linear regression model from scratch
# 
# The weights and biases (`w11, w12,... w23, b1 & b2`) can also be represented as matrices, initialized as random values. The first row of `w` and the first element of `b` are used to predict the first target variable, i.e., yield of apples, and similarly, the second for oranges.

# In[6]:


# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print(w)
print(b)


# `torch.randn` creates a tensor with the given shape, with elements picked randomly from a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) with mean 0 and standard deviation 1.
# 
# Our *model* is simply a function that performs a matrix multiplication of the `inputs` and the weights `w` (transposed) and adds the bias `b` (replicated for each observation).
# 
# ![matrix-mult](https://i.imgur.com/WGXLFvA.png)
# 
# We can define the model as follows:

# In[7]:


def model(x):
    return x @ w.t() + b


# `@` represents matrix multiplication in PyTorch, and the `.t` method returns the transpose of a tensor.
# 
# The matrix obtained by passing the input data into the model is a set of predictions for the target variables.

# In[8]:


# Generate predictions
preds = model(inputs)
print(preds)


# Let's compare the predictions of our model with the actual targets.

# In[9]:


# Compare with targets
print(targets)


# You can see a big difference between our model's predictions and the actual targets because we've initialized our model with random weights and biases. Obviously, we can't expect a randomly initialized model to *just work*.

# ## Loss function
# 
# Before we improve our model, we need a way to evaluate how well our model is performing. We can compare the model's predictions with the actual targets using the following method:
# 
# * Calculate the difference between the two matrices (`preds` and `targets`).
# * Square all elements of the difference matrix to remove negative values.
# * Calculate the average of the elements in the resulting matrix.
# 
# The result is a single number, known as the **mean squared error** (MSE).

# In[10]:


# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()


# `torch.sum` returns the sum of all the elements in a tensor. The `.numel` method of a tensor returns the number of elements in a tensor. Let's compute the mean squared error for the current predictions of our model.

# In[11]:


# Compute loss
loss = mse(preds, targets)
print(loss)


# Here’s how we can interpret the result: *On average, each element in the prediction differs from the actual target by the square root of the loss*. And that’s pretty bad, considering the numbers we are trying to predict are themselves in the range 50–200. The result is called the *loss* because it indicates how bad the model is at predicting the target variables. It represents information loss in the model: the lower the loss, the better the model.

# ## Compute gradients
# 
# With PyTorch, we can automatically compute the gradient or derivative of the loss w.r.t. to the weights and biases because they have `requires_grad` set to `True`. We'll see how this is useful in just a moment.

# In[12]:


# Compute gradients
loss.backward()


# The gradients are stored in the `.grad` property of the respective tensors. Note that the derivative of the loss w.r.t. the weights matrix is itself a matrix with the same dimensions.

# In[13]:


# Gradients for weights
print(w)
print(w.grad)


# ## Adjust weights and biases to reduce the loss
# 
# The loss is a [quadratic function](https://en.wikipedia.org/wiki/Quadratic_function) of our weights and biases, and our objective is to find the set of weights where the loss is the lowest. If we plot a graph of the loss w.r.t any individual weight or bias element, it will look like the figure shown below. An important insight from calculus is that the gradient indicates the rate of change of the loss, i.e., the loss function's [slope](https://en.wikipedia.org/wiki/Slope) w.r.t. the weights and biases.
# 
# If a gradient element is **positive**:
# 
# * **increasing** the weight element's value slightly will **increase** the loss
# * **decreasing** the weight element's value slightly will **decrease** the loss
# 
# ![postive-gradient](https://i.imgur.com/WLzJ4xP.png)
# 
# If a gradient element is **negative**:
# 
# * **increasing** the weight element's value slightly will **decrease** the loss
# * **decreasing** the weight element's value slightly will **increase** the loss
# 
# ![negative=gradient](https://i.imgur.com/dvG2fxU.png)
# 
# The increase or decrease in the loss by changing a weight element is proportional to the gradient of the loss w.r.t. that element. This observation forms the basis of _the gradient descent_ optimization algorithm that we'll use to improve our model (by _descending_ along the _gradient_).
# 
# We can subtract from each weight element a small quantity proportional to the derivative of the loss w.r.t. that element to reduce the loss slightly.

# In[14]:


w
w.grad


# In[15]:


with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5


# We multiply the gradients with a very small number (`10^-5` in this case) to ensure that we don't modify the weights by a very large amount. We want to take a small step in the downhill direction of the gradient, not a giant leap. This number is called the *learning rate* of the algorithm. 
# 
# We use `torch.no_grad` to indicate to PyTorch that we shouldn't track, calculate, or modify gradients while updating the weights and biases.

# In[16]:


# Let's verify that the loss is actually lower
loss = mse(preds, targets)
print(loss)


# Before we proceed, we reset the gradients to zero by invoking the `.zero_()` method. We need to do this because PyTorch accumulates gradients. Otherwise, the next time we invoke `.backward` on the loss, the new gradient values are added to the existing gradients, which may lead to unexpected results.

# In[17]:


w.grad.zero_()
b.grad.zero_()
print(w.grad)
print(b.grad)


# ## Train the model using gradient descent
# 
# As seen above, we reduce the loss and improve our model using the gradient descent optimization algorithm. Thus, we can _train_ the model using the following steps:
# 
# 1. Generate predictions
# 
# 2. Calculate the loss
# 
# 3. Compute gradients w.r.t the weights and biases
# 
# 4. Adjust the weights by subtracting a small quantity proportional to the gradient
# 
# 5. Reset the gradients to zero
# 
# Let's implement the above step by step.

# In[18]:


# Generate predictions
preds = model(inputs)
print(preds)


# In[19]:


# Calculate the loss
loss = mse(preds, targets)
print(loss)


# In[20]:


# Compute gradients
loss.backward()
print(w.grad)
print(b.grad)


# Let's update the weights and biases using the gradients computed above.

# In[21]:


# Adjust weights & reset gradients
with torch.no_grad():
    w -= w.grad * 1e-5
    b -= b.grad * 1e-5
    w.grad.zero_()
    b.grad.zero_()


# Let's take a look at the new weights and biases.

# In[22]:


print(w)
print(b)


# With the new weights and biases, the model should have a lower loss.

# In[23]:


# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print(loss)


# We have already achieved a significant reduction in the loss merely by adjusting the weights and biases slightly using gradient descent.

# ## Train for multiple epochs
# 
# To reduce the loss further, we can repeat the process of adjusting the weights and biases using the gradients multiple times. Each iteration is called an _epoch_. Let's train the model for 100 epochs.

# In[53]:


# Train for 100 epochs
for i in range(500):
    preds = model(inputs)
    loss = mse(preds, targets)
    loss.backward()
    with torch.no_grad():
        w -= w.grad * 1e-5
        b -= b.grad * 1e-5
        w.grad.zero_()
        b.grad.zero_()


# Once again, let's verify that the loss is now lower:

# In[25]:


# Calculate loss
preds = model(inputs)
loss = mse(preds, targets)
print("loss \/")
print(loss)


# The loss is now much lower than its initial value. Let's look at the model's predictions and compare them with the targets.

# In[54]:


# Predictions
print(preds)


# In[27]:


# Targets
print(targets)