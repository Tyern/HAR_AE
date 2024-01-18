#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os

import numpy as np
import pandas as pd

import torch
import torchvision

import matplotlib.pyplot as plt
import lightning as L

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from sklearn.model_selection import train_test_split


# In[2]:


TEST = True

random_seed = 42
L.seed_everything(random_seed)


# In[3]:


TRAIN_SIZE, VAL_SIZE, TEST_SIZE = 0.8, 0.1, 0.1


# ## TRAIN TEST SPLIT

# In[4]:


fft_file = "dataset/processed_data/torso_fft.npy"
label_file = "dataset/processed_data/torso_label.npy"


# In[5]:


fft_train_file = "dataset/processed_data/torso_train_fft.npy"
fft_val_file = "dataset/processed_data/torso_val_fft.npy"
fft_test_file = "dataset/processed_data/torso_test_fft.npy"

label_train_file = "dataset/processed_data/torso_train_label.npy"
label_val_file = "dataset/processed_data/torso_val_label.npy"
label_test_file = "dataset/processed_data/torso_test_label.npy"


# In[6]:


fft_data = np.load(fft_file)
label_data = np.load(label_file)


# In[7]:


label_data.shape


# In[9]:


activity_range = list(range(1, 8+1))

label_idx = np.array([not any(x - x[0]) for x in label_data])
print("label_idx", label_idx)


# In[10]:


data_filtered = fft_data[label_idx]
label_filtered = label_data[label_idx]


# In[11]:


print("data_filtered.shape", data_filtered.shape)
print("label_filtered.shape", label_filtered.shape)


# In[12]:


label = label_filtered[:, 0]
label


# In[13]:


train_val_data, test_data, train_val_label, test_label = \
    train_test_split(data_filtered, label, test_size=TEST_SIZE, stratify=label, shuffle=True)

train_data, val_data, train_label, val_label = \
    train_test_split(train_val_data, train_val_label, test_size=VAL_SIZE / (TRAIN_SIZE + VAL_SIZE), stratify=train_val_label, shuffle=True)

print("train_data.shape, train_label.shape", train_data.shape, train_label.shape)
print("val_data.shape, val_label.shape", val_data.shape, val_label.shape)
print("test_data.shape, test_label.shape", test_data.shape, test_label.shape)


        


# In[14]:


np.save(fft_train_file, train_data)
np.save(label_train_file, train_label)
np.save(fft_val_file, val_data)
np.save(label_val_file, val_label)
np.save(fft_test_file, test_data)
np.save(label_test_file, test_label)


# In[ ]:




