#!/usr/bin/env python
# coding: utf-8

# # Human Faces - CelebA GAN
# 
# Make Your First GAN With PyTorch, 2020

# In[ ]:


# mount Drive to access data files

from google.colab import drive
drive.mount('./mount')


# In[ ]:


# import libraries

import torch
import torch.nn as nn
from torch.utils.data import Dataset

import h5py
import pandas, numpy, random
import matplotlib.pyplot as plt


# ## Standard CUDA Check And Set Up

# In[ ]:


# check if CUDA is available
# if yes, set default tensor type to cuda

if torch.cuda.is_available():
  torch.set_default_tensor_type(torch.cuda.FloatTensor)
  print("using cuda:", torch.cuda.get_device_name(0))
  pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

device


# ## Dataset Class

# In[ ]:


# dataset class

class CelebADataset(Dataset):
    
    def __init__(self, file):
        self.file_object = h5py.File(file, 'r')
        self.dataset = self.file_object['img_align_celeba']
        pass
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, index):
        if (index >= len(self.dataset)):
          raise IndexError()
        img = numpy.array(self.dataset[str(index)+'.jpg'])
        return torch.cuda.FloatTensor(img) / 255.0
    
    def plot_image(self, index):
        plt.imshow(numpy.array(self.dataset[str(index)+'.jpg']), interpolation='nearest')
        pass
    
    pass


# In[ ]:


# create Dataset object

celeba_dataset = CelebADataset('mount/My Drive/Colab Notebooks/myo_gan/celeba_dataset/celeba_aligned_small.h5py')


# In[ ]:


# check data contains images

celeba_dataset.plot_image(43)


# # Helper Functions

# In[ ]:


# functions to generate random data

def generate_random_image(size):
    random_data = torch.rand(size)
    return random_data


def generate_random_seed(size):
    random_data = torch.randn(size)
    return random_data


# In[ ]:


# modified from https://github.com/pytorch/vision/issues/720

class View(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape,

    def forward(self, x):
        return x.view(*self.shape)


# ## Discriminator Network

# In[ ]:


# discriminator class

class Discriminator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            View(218*178*3),
            
            nn.Linear(3*218*178, 100),
            nn.LeakyReLU(),
            
            nn.LayerNorm(100),
            
            nn.Linear(100, 1),
            nn.Sigmoid()
        )
        
        # create loss function
        self.loss_function = nn.BCELoss()

        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
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
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass
        if (self.counter % 1000 == 0):
            print("counter = ", self.counter)
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass


# ## Test Discriminator

# In[ ]:


get_ipython().run_cell_magic('time', '', '# test discriminator can separate real data from random noise\n\nD = Discriminator()\n# move model to cuda device\nD.to(device)\n\nfor image_data_tensor in celeba_dataset:\n    # real data\n    D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))\n    # fake data\n    D.train(generate_random_image((218,178,3)), torch.cuda.FloatTensor([0.0]))\n    pass')


# In[ ]:


# plot discriminator loss

D.plot_progress()


# In[ ]:


# manually run discriminator to check it can tell real data from fake

for i in range(4):
  image_data_tensor = celeba_dataset[random.randint(0,20000)]
  print( D.forward( image_data_tensor ).item() )
  pass

for i in range(4):
  print( D.forward( generate_random_image((218,178,3))).item() )
  pass


# ## Generator Network

# In[ ]:


# generator class

class Generator(nn.Module):
    
    def __init__(self):
        # initialise parent pytorch class
        super().__init__()
        
        # define neural network layers
        self.model = nn.Sequential(
            nn.Linear(100, 3*10*10),
            nn.LeakyReLU(),
            
            nn.LayerNorm(3*10*10),
            
            nn.Linear(3*10*10, 3*218*178),
            
            nn.Sigmoid(),
            View((218,178,3))
        )
        
        # create optimiser, simple stochastic gradient descent
        self.optimiser = torch.optim.Adam(self.parameters(), lr=0.0001)

        # counter and accumulator for progress
        self.counter = 0;
        self.progress = []
        
        pass
    
    
    def forward(self, inputs):        
        # simply run model
        return self.model(inputs)
    
    
    def train(self, D, inputs, targets):
        # calculate the output of the network
        g_output = self.forward(inputs)
        
        # pass onto Discriminator
        d_output = D.forward(g_output)
        
        # calculate error
        loss = D.loss_function(d_output, targets)

        # increase counter and accumulate error every 10
        self.counter += 1;
        if (self.counter % 10 == 0):
            self.progress.append(loss.item())
            pass

        # zero gradients, perform a backward pass, update weights
        self.optimiser.zero_grad()
        loss.backward()
        self.optimiser.step()

        pass
    
    
    def plot_progress(self):
        df = pandas.DataFrame(self.progress, columns=['loss'])
        df.plot(ylim=(0), figsize=(16,8), alpha=0.1, marker='.', grid=True, yticks=(0, 0.25, 0.5, 1.0, 5.0))
        pass
    
    pass


# ## Test Generator Output

# In[ ]:


# check the generator output is of the right type and shape

G = Generator()
# move model to cuda device
G.to(device)

output = G.forward(generate_random_seed(100))

img = output.detach().cpu().numpy()

plt.imshow(img, interpolation='none', cmap='Blues')


# ## Train GAN

# In[ ]:


get_ipython().run_cell_magic('time', '', '\n# create Discriminator and Generator\n\nD = Discriminator()\nD.to(device)\nG = Generator()\nG.to(device)\n\nepochs = 1\n\nfor epoch in range(epochs):\n  print ("epoch = ", epoch + 1)\n\n  # train Discriminator and Generator\n\n  for image_data_tensor in celeba_dataset:\n    # train discriminator on true\n    D.train(image_data_tensor, torch.cuda.FloatTensor([1.0]))\n    \n    # train discriminator on false\n    # use detach() so gradients in G are not calculated\n    D.train(G.forward(generate_random_seed(100)).detach(), torch.cuda.FloatTensor([0.0]))\n    \n    # train generator\n    G.train(D, generate_random_seed(100), torch.cuda.FloatTensor([1.0]))\n\n    pass\n    \n  pass')


# In[ ]:


# plot discriminator error

D.plot_progress()


# In[ ]:


# plot generator error

G.plot_progress()


# ## Run Generator

# In[ ]:


# plot several outputs from the trained generator

# plot a 3 column, 2 row array of generated images
f, axarr = plt.subplots(2,3, figsize=(16,8))
for i in range(2):
    for j in range(3):
        output = G.forward(generate_random_seed(100))
        img = output.detach().cpu().numpy()
        axarr[i,j].imshow(img, interpolation='none', cmap='Blues')
        pass
    pass


# # Memory Consumption

# In[ ]:


# current memory allocated to tensors (in Gb)

torch.cuda.memory_allocated(device) / (1024*1024*1024)


# In[ ]:


# total memory allocated to tensors during program (in Gb)

torch.cuda.max_memory_allocated(device) / (1024*1024*1024)


# In[ ]:


# summary of memory consumption

print(torch.cuda.memory_summary(device, abbreviated=True))


# In[ ]:




