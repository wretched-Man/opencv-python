#!/usr/bin/env python
# coding: utf-8

# # Color Segmentation

# In this notebook we will use the HSV color space to segment images based on color.

# In[1]:


import numpy as np
import cv2
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'


# We will use a color dictionary that will enable us to choose the colors that we want

# In[2]:


color_dict_HSV = {'black': [[180, 255, 30], [0, 0, 0]],
              'white': [[180, 18, 255], [0, 0, 231]],
              'red1': [[180, 255, 255], [159, 50, 70]],
              'red2': [[9, 255, 255], [0, 50, 70]],
              'green': [[89, 255, 255], [36, 50, 70]],
              'blue': [[128, 255, 255], [90, 50, 70]],
              'yellow': [[35, 255, 255], [25, 50, 70]],
              'purple': [[158, 255, 255], [129, 50, 70]],
              'orange': [[24, 255, 255], [10, 50, 70]],
              'gray': [[180, 18, 230], [0, 0, 40]]}


# Let us now take a look at examples of segmentation.

# ## First Example

# In[3]:


cat_alley = cv2.imread('images/cat_alley_segment.jpg', cv2.IMREAD_COLOR)

plt.figure(figsize=[10,5])
plt.imshow(cat_alley[:, :, ::-1])


# We are going to segment the image above (most of it atleast).
# To do this, we will use cv2.inRange function with the HSV of this image.
# It returns a mask (values 0 & 255).

# Just a note: Hue is about identifying the core color.
# It refers to the dominant color family of a specific color.
# It’s the underlying base color of the mixture you’re looking at

# In[4]:


#convert to hsv
cat_hsv = cv2.cvtColor(cat_alley, cv2.COLOR_BGR2HSV)


# In[5]:


#returns a mask
def mask_builder(hsv, hue):
    lower = np.array(color_dict_HSV[hue][1])
    upper = np.array(color_dict_HSV[hue][0])

    return cv2.inRange(hsv, lower, upper) #mask


# The cat image contains many different hues so we will create various masks

# In[6]:


red1_mask = mask_builder(cat_hsv, 'red1')
red2_mask = mask_builder(cat_hsv, 'red2')
orange_mask = mask_builder(cat_hsv, 'orange')


# In[7]:


#A combination of reds
reds = cv2.bitwise_or(red1_mask, red2_mask)

#A combination of reds and orange
red_oranges = cv2.bitwise_or(orange_mask, reds)


# In[8]:


#This is what we have as our final mask
plt.figure(figsize=[10, 5])
plt.imshow(red_oranges)


# In[9]:


#Now to build our image
#We and the BGR image with itself and apply the inRange mask
cat_segment = cv2.bitwise_and(cat_alley, cat_alley, mask=red_oranges)

plt.figure(figsize=[10, 5])
plt.imshow(cat_segment[:, :, ::-1])


# Wow! Our cat is missing 3 legs and an eye... Oh well!
# We can also do this by using the Hue channel.

# In[10]:


#We will take the hue channel and threshold it,
#making it a mask
_, hue_mask = cv2.threshold(cat_hsv[:, :, 0], 50, 255, cv2.THRESH_BINARY_INV)

plt.figure(figsize=[10, 5])
plt.subplot(121); plt.imshow(cat_hsv[:, :, 0]); plt.title('Hue Channel')
plt.subplot(122); plt.imshow(hue_mask); plt.title('Hue Mask')


# We see that the hue mask will give us a better image. We now use it.

# In[11]:


cat_hue_segment = cv2.bitwise_and(cat_alley, cat_alley, mask=hue_mask)

plt.figure(figsize=[10, 5])
plt.imshow(cat_hue_segment[:, :, ::-1])


# We see we now have a better image.

# ## Second Example

# In[12]:


#We will now try a second example... a flower
flower_plus = cv2.imread('images/red_hibiscus.jpg')

plt.figure(figsize=[10, 5])
plt.imshow(flower_plus[:, :, ::-1])


# Our aim is to segment the flower portion only.
# For that we need to know the hues that make it up.

# In[13]:


#We convert to hsv
flower_hsv = cv2.cvtColor(flower_plus, cv2.COLOR_BGR2HSV)


# In[14]:


#Here I only build the red1 and orange color masks
#However, prevoiusly I had built multiple and displayed them
#to see which primary colors exist in the image
red1_mask = mask_builder(flower_hsv, 'red1')
red2_mask = mask_builder(flower_hsv, 'red2')
orange_mask = mask_builder(flower_hsv, 'orange')


# In[15]:


#Displaying the acceptable colors
#The three colors here give us the most acceptable range,
#and,combined, give us the mask we desire
plt.figure(figsize=[12, 4])
plt.subplot(131); plt.imshow(red1_mask); plt.title('Red1 Mask')
plt.subplot(132); plt.imshow(red2_mask); plt.title('Red2 Mask')
plt.subplot(133); plt.imshow(orange_mask); plt.title('Orange Mask')


# In[16]:


#Mixing the masks into a final mask
reds = cv2.bitwise_or(red1_mask, red2_mask)
reds_oranges = cv2.bitwise_or(reds, orange_mask)


# In[17]:


#Displaying our final mask
plt.figure()
plt.imshow(reds_oranges)


# In[19]:


#We will now apply the mask to the image
hibiscus = cv2.bitwise_and(flower_plus, flower_plus, mask=reds_oranges)

#displaying the flower
plt.figure()
plt.imshow(hibiscus[:, :, ::-1])


# We now have our image. We can now save it.

# In[21]:


cv2.imwrite('red_hibiscus_segmented.jpg', hibiscus)

