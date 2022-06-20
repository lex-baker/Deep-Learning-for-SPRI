# ## Linear regression using PyTorch built-ins
# 
# We've implemented linear regression & gradient descent model using some basic tensor operations. However, since this is a common pattern in deep learning, PyTorch provides several built-in functions and classes to make it easy to create and train models with just a few lines of code.
# 
# Let's begin by importing the `torch.nn` package from PyTorch, which contains utility classes for building neural networks.

import torch
import numpy as np

# In[31]:


import torch.nn as nn


# As before, we represent the inputs and targets and matrices.

# In[32]:


# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43], 
                   [91, 88, 64], 
                   [87, 134, 58], 
                   [102, 43, 37], 
                   [69, 96, 70], 
                   [74, 66, 43], 
                   [91, 87, 65], 
                   [88, 134, 59], 
                   [101, 44, 37], 
                   [68, 96, 71], 
                   [73, 66, 44], 
                   [92, 87, 64], 
                   [87, 135, 57], 
                   [103, 43, 36], 
                   [68, 97, 70]], 
                  dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70], 
                    [81, 101], 
                    [119, 133], 
                    [22, 37], 
                    [103, 119],
                    [57, 69], 
                    [80, 102], 
                    [118, 132], 
                    [21, 38], 
                    [104, 118], 
                    [57, 69], 
                    [82, 100], 
                    [118, 134], 
                    [20, 38], 
                    [102, 120]], 
                   dtype='float32')

inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)


# In[33]:

print("inputs")
print(inputs)


# We are using 15 training examples to illustrate how to work with large datasets in small batches. 

# ## Dataset and DataLoader
# 
# We'll create a `TensorDataset`, which allows access to rows from `inputs` and `targets` as tuples, and provides standard APIs for working with many different types of datasets in PyTorch.

# In[34]:


from torch.utils.data import TensorDataset


# In[35]:


# Define dataset
train_ds = TensorDataset(inputs, targets)
train_ds[0:3]


# The `TensorDataset` allows us to access a small section of the training data using the array indexing notation (`[0:3]` in the above code). It returns a tuple with two elements. The first element contains the input variables for the selected rows, and the second contains the targets.

# We'll also create a `DataLoader`, which can split the data into batches of a predefined size while training. It also provides other utilities like shuffling and random sampling of the data.

# In[36]:


from torch.utils.data import DataLoader


# In[37]:


# Define data loader
batch_size = 5
train_dl = DataLoader(train_ds, batch_size, shuffle=True)


# We can use the data loader in a `for` loop. Let's look at an example.

# In[38]:


for xb, yb in train_dl:
    print(xb)
    print(yb)
    break


# In each iteration, the data loader returns one batch of data with the given batch size. If `shuffle` is set to `True`, it shuffles the training data before creating batches. Shuffling helps randomize the input to the optimization algorithm, leading to a faster reduction in the loss.

# ## nn.Linear
# 
# Instead of initializing the weights & biases manually, we can define the model using the `nn.Linear` class from PyTorch, which does it automatically.

# In[39]:


# Define model
model = nn.Linear(3, 2)
print(model.weight)
print(model.bias)


# PyTorch models also have a helpful `.parameters` method, which returns a list containing all the weights and bias matrices present in the model. For our linear regression model, we have one weight matrix and one bias matrix.

# In[40]:


# Parameters
list(model.parameters())


# We can use the model to generate predictions in the same way as before.

# In[41]:


# Generate predictions
preds = model(inputs)
preds


# ## Loss Function
# 
# Instead of defining a loss function manually, we can use the built-in loss function `mse_loss`.

# In[42]:


# Import nn.functional
import torch.nn.functional as F


# The `nn.functional` package contains many useful loss functions and several other utilities. 

# In[43]:


# Define loss function
loss_fn = F.mse_loss


# Let's compute the loss for the current predictions of our model.

# In[44]:


loss = loss_fn(model(inputs), targets)
print(loss)


# ## Optimizer
# 
# Instead of manually manipulating the model's weights & biases using gradients, we can use the optimizer `optim.SGD`. SGD is short for "stochastic gradient descent". The term _stochastic_ indicates that samples are selected in random batches instead of as a single group.

# In[45]:


# Define optimizer
opt = torch.optim.SGD(model.parameters(), lr=1e-5)


# Note that `model.parameters()` is passed as an argument to `optim.SGD` so that the optimizer knows which matrices should be modified during the update step. Also, we can specify a learning rate that controls the amount by which the parameters are modified.

# ## Train the model
# 
# The only change is that we'll work batches of data instead of processing the entire training data in every iteration. Let's define a utility function `fit` that trains the model for a given number of epochs.

# In[46]:


# Utility function to train the model
def fit(num_epochs, model, loss_fn, opt, train_dl):
    
    # Repeat for given number of epochs
    for epoch in range(num_epochs):
        
        # Train with batches of data
        for xb,yb in train_dl:
            
            # 1. Generate predictions
            pred = model(xb)
            
            # 2. Calculate loss
            loss = loss_fn(pred, yb)
            
            # 3. Compute gradients
            loss.backward()
            
            # 4. Update parameters using gradients
            opt.step()
            
            # 5. Reset the gradients to zero
            opt.zero_grad()
        
        # Print the progress
        if (epoch+1) % 10 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))


# Some things to note above:
# 
# * We use the data loader defined earlier to get batches of data for every iteration.
# 
# * Instead of updating parameters (weights and biases) manually, we use `opt.step` to perform the update and `opt.zero_grad` to reset the gradients to zero.
# 
# * We've also added a log statement that prints the loss from the last batch of data for every 10th epoch to track training progress. `loss.item` returns the actual value stored in the loss tensor.
# 
# Let's train the model for 100 epochs.

# In[47]:


fit(500, model, loss_fn, opt, train_dl)


# Let's generate predictions using our model and verify that they're close to our targets.

# In[48]:


# Generate predictions
preds = model(inputs)
print("final preds")
print(preds)


# In[49]:


# Compare with targets
print("targets")
print(targets)


# Indeed, the predictions are quite close to our targets. We have a trained a reasonably good model to predict crop yields for apples and oranges by looking at the average temperature, rainfall, and humidity in a region. We can use it to make predictions of crop yields for new regions by passing a batch containing a single row of input.

# In[50]:


print(model(torch.tensor([[75, 63, 44.]])))


# The predicted yield of apples is 54.3 tons per hectare, and that of oranges is 68.3 tons per hectare.

# ## Machine Learning vs. Classical Programming
# 
# The approach we've taken in this tutorial is very different from programming as you might know it. Usually, we write programs that take some inputs, perform some operations, and return a result. 
# 
# However, in this notebook, we've defined a "model" that assumes a specific relationship between the inputs and the outputs, expressed using some unknown parameters (weights & biases). We then show the model some know inputs and outputs and _train_ the model to come up with good values for the unknown parameters. Once trained, the model can be used to compute the outputs for new inputs.
# 
# This paradigm of programming is known as _machine learning_, where we use data to figure out the relationship between inputs and outputs. _Deep learning_ is a branch of machine learning that uses matrix operations, non-linear activation functions and gradient descent to build and train models. Andrej Karpathy, the director of AI at Tesla Motors, has written a great blog post on this topics, titled [Software 2.0](https://medium.com/@karpathy/software-2-0-a64152b37c35).
# 
# This picture from book [Deep Learning with Python](https://www.manning.com/books/deep-learning-with-python) by Francois Chollet captures the difference between classical programming and machine learning:
# 
# ![](https://i.imgur.com/oJEQe7k.png)
# 
# Keep this picture in mind as you work through the next few tutorials. 
# 
# 

# ## Commit and update the notebook
# 
# As a final step, we can record a new version of the notebook using the `jovian` library.

# In[51]:


import jovian


# In[52]:


jovian.commit(project='02-linear-regression')


# Note that running `jovian.commit` a second time records a new version of your existing notebook. With Jovian.ml, you can avoid creating copies of your Jupyter notebooks and keep versions organized. Jovian also provides a visual diff ([example](https://jovian.ai/aakashns/keras-mnist-jovian/diff?base=8&remote=2)) so you can inspect what has changed between different versions:
# 
# ![jovian-commenting](https://i.imgur.com/HF1cOVt.png)

# ## Exercises and Further Reading
# 
# We've covered the following topics in this tutorial:
# 
# - Introduction to linear regression and gradient descent
# - Implementing a linear regression model using PyTorch tensors
# - Training a linear regression model using the gradient descent algorithm
# - Implementing gradient descent and linear regression using PyTorch built-in
# 
# 
# Here are some resources for learning more about linear regression and gradient descent:
# 
# * An visual & animated explanation of gradient descent: https://www.youtube.com/watch?v=IHZwWFHWa-w
# 
# * For a more detailed explanation of derivates and gradient descent, see [these notes from a Udacity course](https://storage.googleapis.com/supplemental_media/udacityu/315142919/Gradient%20Descent.pdf). 
# 
# * For an animated visualization of how linear regression works, [see this post](https://hackernoon.com/visualizing-linear-regression-with-pytorch-9261f49edb09).
# 
# * For a more mathematical treatment of matrix calculus, linear regression and gradient descent, you should check out [Andrew Ng's excellent course notes](https://github.com/Cleo-Stanford-CS/CS229_Notes/blob/master/lectures/cs229-notes1.pdf) from CS229 at Stanford University.
# 
# * To practice and test your skills, you can participate in the [Boston Housing Price Prediction](https://www.kaggle.com/c/boston-housing) competition on Kaggle, a website that hosts data science competitions.
# 
# With this, we complete our discussion of linear regression in PyTorch, and we’re ready to move on to the next topic: [Working with Images & Logistic Regression](https://jovian.ai/aakashns/03-logistic-regression).

# ## Questions for Review
# 
# Try answering the following questions to test your understanding of the topics covered in this notebook:
# 
# 1. What is a linear regression model? Give an example of a problem formulated as a linear regression model.
# 2. What are input and target variables in a dataset? Give an example.
# 3. What are weights and biases in a linear regression model?
# 4. How do you represent tabular data using PyTorch tensors?
# 5. Why do we create separate matrices for inputs and targets while training a linear regression model?
# 6. How do you determine the shape of the weights matrix & bias vector given some training data?
# 7. How do you create randomly initialized weights & biases with a given shape?
# 8. How is a linear regression model implemented using matrix operations? Explain with an example.
# 9. How do you generate predictions using a linear regression model?
# 10. Why are the predictions of a randomly initialized model different from the actual targets?
# 11. What is a loss function? What does the term “loss” signify?
# 12. What is mean squared error?
# 13. Write a function to calculate mean squared using model predictions and actual targets.
# 14. What happens when you invoke the `.backward` function on the result of the mean squared error loss function?
# 15. Why is the derivative of the loss w.r.t. the weights matrix itself a matrix? What do its elements represent?
# 16. How is the derivate of the loss w.r.t. a weight element useful for reducing the loss? Explain with an example.
# 17. Suppose the derivative  of the loss w.r.t. a weight element is positive. Should you increase or decrease the element’s value slightly to get a lower loss?
# 18. Suppose the derivative  of the loss w.r.t. a weight element is negative. Should you increase or decrease the element’s value slightly to get a lower loss?
# 19. How do you update the weights and biases of a model using their respective gradients to reduce the loss slightly?
# 20. What is the gradient descent optimization algorithm? Why is it called “gradient descent”?
# 21. Why do you subtract a “small quantity” proportional to the gradient from the weights & biases, not the actual gradient itself?
# 22. What is learning rate? Why is it important?
# 23. What is `torch.no_grad`?
# 24. Why do you reset gradients to zero after updating weights and biases?
# 25. What are the steps involved in training a linear regression model using gradient descent?
# 26. What is an epoch?
# 27. What is the benefit of training a model for multiple epochs?
# 28. How do you make predictions using a trained model?
# 29. What should you do if your model’s loss doesn’t decrease while training? Hint: learning rate.
# 30. What is `torch.nn`?
# 31. What is the purpose of the `TensorDataset` class in PyTorch? Give an example.
# 32. What is a data loader in PyTorch? Give an example.
# 33. How do you use a data loader to retrieve batches of data?
# 34. What are the benefits of shuffling the training data before creating batches?
# 35. What is the benefit of training in small batches instead of training with the entire dataset?
# 36. What is the purpose of the `nn.Linear` class in PyTorch? Give an example.
# 37. How do you see the weights and biases of a `nn.Linear` model?
# 38. What is the purpose of the `torch.nn.functional` module?
# 39. How do you compute mean squared error loss using a PyTorch built-in function?
# 40. What is an optimizer in PyTorch?
# 41. What is `torch.optim.SGD`? What does SGD stand for?
# 42. What are the inputs to a PyTorch optimizer? 
# 43. Give an example of creating an optimizer for training a linear regression model.
# 44. Write a function to train a `nn.Linear` model in batches using gradient descent.
# 45. How do you use a linear regression model to make predictions on previously unseen data?
# 