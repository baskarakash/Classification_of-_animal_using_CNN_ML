#!/usr/bin/env python
# coding: utf-8

# In[4]:


#import pakages 
import tensorflow as tf
import numpy as np
import os
import sys
import cv2
import matplotlib.pyplot as plt
import pickle
import random
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Dropout,Activation, Flatten,Conv2D,MaxPooling2D
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#importing data set
DATA_DIR = r"C:\Users\Akash Baskar\Downloads\archive (5)\afhq\val"
c = ["cat","dog"]
IMAGE_SIZE = 30


# In[6]:


#creating a training data set

def create_training_data():
    training_date = []
    for categories in c:
        path = os.path.join(DATA_DIR,categories)
        class_num = c.index(categories)
        for img in os.listdir(path):
            try:
                img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
                training_date.append([new_array,class_num])
            except:
                pass
    return training_date


# In[7]:


data = np.asarray(create_training_data())


# In[8]:


x_data = []
y_data = []

for x in data:
    x_data.append(x[0])
    y_data.append(x[1])


# In[9]:


x_data_np = np.asarray(x_data)/255.0
y_data_np = np.asarray(y_data)


# In[10]:


pickle_out = open('x_data_np','wb')
pickle.dump(x_data_np,pickle_out)
pickle_out.close()


# In[11]:


pickle_out = open('y_data_np','wb')
pickle.dump(y_data_np,pickle_out)
pickle_out.close()


# In[12]:


X_Temp = open('x_data_np','rb')
x_data_np = pickle.load(X_Temp)

Y_Temp = open('y_data_np','rb')
y_data_np = pickle.load(Y_Temp)


# In[13]:


x_data_np = x_data_np.reshape(-1, 30, 30, 1)


# In[14]:


#spliting the data set for training and testing
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train, Y_Test = train_test_split(x_data_np, y_data_np, test_size=0.3,random_state=101)


# In[15]:


#implementing CNN model
model = Sequential()
model.add(Conv2D(150, (3, 3), input_shape=x_data_np.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(75, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


# In[16]:


#training the data set
model.fit(x_data_np, y_data_np, batch_size=32, epochs=1, validation_split=0.3)
model.save('64x3CNN.model')


# In[17]:


def prepare(filepath):
    training_date = []
    
    img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array,(IMAGE_SIZE,IMAGE_SIZE))
    new_image =  new_array.reshape(-1,IMAGE_SIZE,IMAGE_SIZE,1)
    return new_image


# In[18]:


model = tf.keras.models.load_model('64x3CNN.model')


# In[22]:


#importing the image to get the output
filepath = 'C:/Users/Akash Baskar/Desktop/cat.jpg'
img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)

plt.imshow(img_array)


# In[20]:


test = model.predict([prepare(filepath=filepath )])


# In[21]:


print(c[int(test[0][0])])

