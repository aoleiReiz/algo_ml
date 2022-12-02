#!/usr/bin/env python
# coding: utf-8

# # First PyTorch Neural Network - MNIST Classifier
# 
# Make Your First GAN With PyTorch, 2020

# In[1]:


get_ipython().system('nvidia-smi')


# In[2]:


# import libraries

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import pandas
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=UserWarning)


# ## DataSet Class

# In[3]:


# dataset class

class MnistDataset(Dataset):
    
    def __init__(self, csv_file):
        self.data_df = pandas.read_csv(csv_file, header=None)
        pass
    
    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, index):
        # image target (label)
        label = self.data_df.iloc[index,0]
        target = torch.zeros((10))
        target[label] = 1.0
        
        # image data, normalised from 0-255 to 0-1
        image_values = torch.FloatTensor(self.data_df.iloc[index,1:].values) / 255.0
        
        # return label, image data tensor and target tensor
        return label, image_values, target
    
    def plot_image(self, index):
        img = self.data_df.iloc[index,1:].values.reshape(28,28)
        plt.title("label = " + str(self.data_df.iloc[index,0]))
        plt.imshow(img, interpolation='none', cmap='Blues')
        pass
    
    pass


# ## Load Data

# In[4]:


mnist_dataset = MnistDataset('mnist_data/mnist_train.csv')


# In[5]:


# check data contains images

mnist_dataset.plot_image(9)


# In[6]:


# check Dataset class can be accessed by index, returns label, image values and target tensor

mnist_dataset[100]


# ## Neural Network Class

# In[7]:


# classifier class

class Classifier(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(784, 200),
            nn.Sigmoid(),
            nn.Linear(200, 10),
            nn.Sigmoid()
        )
        
        # create loss function
        self.loss_function = nn.MSELoss()

        # create optimiser, using simple stochastic gradient descent
        self.optimiser = torch.optim.SGD(self.parameters(), lr=0.01)

        # counter and accumulator for progress
        self.counter = 0
        self.progress = []

        pass
    
    
    def forward(self, inputs):
        # simply run model
        return self.model(inputs)
    
    
    def train(self, inputs, targets):
        # calculate the output of the network
        outputs = self.forward(inputs)
        
        # calculate loss
        loss = self.loss_function(outputs, targets)

        # increase counter and accumulate error every 10
        self.counter += 1
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 10000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, and update the weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0, 1.0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5))
        pass
    
    pass


# ## Train Neural Network

# In[ ]:


get_ipython().run_cell_magic('time', '', '# create neural network\n\nC = Classifier()\n\n# train network on MNIST data set\n\nepochs = 4\n\nfor i in range(epochs):\n    print(\'training epoch\', i+1, "of", epochs)\n    for label, image_data_tensor, target_tensor in mnist_dataset:\n        C.train(image_data_tensor, target_tensor)\n        pass\n    pass')


# In[ ]:


# plot classifier error

C.plot_progress()


# In[ ]:




