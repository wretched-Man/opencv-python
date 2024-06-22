#!/usr/bin/env python
# coding: utf-8

# # Color Segmentation 2

# In[209]:


#We will now use a logo
import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'


# In[220]:


#Our color dictionary for HSV ranges
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


# In[226]:


#Create a mask
def mask_builder(hsv, hue):
    lower = np.array(color_dict_HSV[hue][1])
    upper = np.array(color_dict_HSV[hue][0])

    return cv2.inRange(hsv, lower, upper) #mask


# In[227]:


#load the image
logo = cv2.imread('images/google_G_segment.png', cv2.IMREAD_COLOR)

plt.figure(figsize=[5, 5])
plt.imshow(logo[:, :, ::-1])


# We are going to segment the various portions of the logo.

# In[219]:


#we first convert to HSV
logo_hsv = cv2.cvtColor(logo, cv2.COLOR_BGR2HSV)


# In[230]:


#We now build masks for the various hues in the image
green_mask = mask_builder(logo_hsv, 'green')
red_mask = mask_builder(logo_hsv, 'red2')
orange_mask = mask_builder(logo_hsv, 'orange')
blue_mask = mask_builder(logo_hsv, 'blue')


# In[231]:


#Display the masks
plt.figure(figsize=[20, 4])
plt.subplot(141); plt.imshow(red_mask); plt.title('Red Mask')
plt.subplot(142); plt.imshow(orange_mask); plt.title('Orange Mask')
plt.subplot(143); plt.imshow(green_mask); plt.title('Green Mask')
plt.subplot(144); plt.imshow(blue_mask); plt.title('Blue Mask')


# We will now overlay our mask on the original image.

# In[234]:


green = cv2.bitwise_and(logo, logo, mask=green_mask)
red = cv2.bitwise_and(logo, logo, mask=red_mask)
orange = cv2.bitwise_and(logo, logo, mask=orange_mask)
blue = cv2.bitwise_and(logo, logo, mask=blue_mask)


# In[236]:


#Display the segmented portions
plt.figure(figsize=[20, 4])
plt.subplot(141); plt.imshow(red[:, :, ::-1]); plt.title('Red')
plt.subplot(142); plt.imshow(orange[:, :, ::-1]); plt.title('Orange')
plt.subplot(143); plt.imshow(green[:, :, ::-1]); plt.title('Green')
plt.subplot(144); plt.imshow(blue[:, :, ::-1]); plt.title('Blue')


# And there we have it, our segmented image.
