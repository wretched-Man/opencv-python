#!/usr/bin/env python
# coding: utf-8

# Logo Manipulation
# We are going to learn how to manipulate logos, that is, change the background/ foreground.
# For that, we need to understand **logical operations**.

# Logical Operations
# OpenCV also supports bitwise logical operations that are supported in Python.
# They are known as bitwise since they operate on the bits of a number.
# OpenCV provides `bitwise_and`, `bitwise_or`, `bitwise_not` and `bitwise_xor`.
# These operations are useful in the logo manipulation we are going to do.
# In case you are unfamiliar with logical operations in Python,
# here is a good article.(https://www.geeksforgeeks.org/python-bitwise-operators/) to the same.

# Here is what we aim to achieve.
# ![logo image](https://opencv.org/wp-content/uploads/2021/08/c0-m2-logo-manipulation-cr.png)

# In[2]:


#loading
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib


# In[2]:


#load the images
img_main = cv2.imread("images/CR_Logo.png", cv2.IMREAD_COLOR)
plt.figure(figsize=[3, 3])
plt.imshow(img_main[:, :, ::-1])
#Save shape to resize checkerboard, which is smaller
logo_w = img_main.shape[1]
logo_h = img_main.shape[0]


# In[3]:


#load checkerboard
img_check = cv2.imread("images/checkerboard_color.png", cv2.IMREAD_COLOR)
plt.figure(figsize=[3, 3])
plt.imshow(img_check[:, :, ::-1])


# In[4]:


#Resizing the image
img_check = cv2.resize(img_check.copy(), dsize=(logo_w, logo_h), interpolation=cv2.INTER_AREA)
plt.figure(figsize=[3, 3])
plt.imshow(img_check[:, :, ::-1])
print(img_check.shape)


# In[5]:


#similar size
img_check.shape == img_main.shape


# ## The process

# The process of achieving this is as simple as it is intuitive. We will build 2 images.
# 1. **Image1**
# 
# Image1 will contain 2 parts: A `black background` and a `colored foreground`.
# 
# 2. **Image2**
# 
# Image2 will also contain 2 parts: A `colored foreground` and a `black background`.
# 
# As you can see, Image1 is an 'inverse' of Image2.
# As we continue we will see and expain why these two images are necessary.

# ### Image1

# ### 1. Black Background

# We can do this easily by thresholding the image.

# In[6]:


#
_, img1_background = cv2.threshold(cv2.cvtColor(img_main.copy(),\
                                    cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)
plt.figure(figsize=[3, 3])
plt.imshow(img1_background, 'gray')


# We now have Image1's image background. Now for the colored foreground.
# By colored foreground we mean that the logo part is changed from white
# to the checkered background. We do this by applying `img1_background`
# as a mask on the check image. Performing a bitwise AND operation on an
# image with itself returns the image. When the mask is provided,
# the `bitwise_and` operation is true if at that evaluated pixel the value of the mask is non-zero.

# ### 2. Colored foreground

# In[7]:


img1_foreground = cv2.bitwise_and(img_check, img_check, mask=img1_background)
plt.figure(figsize=[3, 3])
plt.imshow(img1_foreground[:, :, ::-1])
img1_foreground.shape


# We now have the Image1.

# ### Image2

# ### 1. Black foreground

# For Image2, we begin by creating the black foreground. i.e. the logo part
# should be made black. We do this simply by inverting the thresholded black foreground of Image1.

# In[8]:


img2_foreground = cv2.bitwise_not(img1_background)
plt.figure(figsize=[3, 3])
plt.imshow(img2_foreground, 'gray')


# ### 2. Colored Background

# We will now create a colored background for the image.
# This will mean taking the green color from the original logo.
# We can do this using the bitwise AND operation as you will see below.

# In[9]:


img2_background = cv2.bitwise_and(img_main, img_main, mask=img2_foreground)
plt.figure(figsize=[3, 3])
plt.imshow(img2_background[:, :, ::-1])
img2_background.shape


# Now we have all the images, let us see them in one grid.

# In[32]:


plt.figure(figsize=[8, 8])
plt.subplot(221); plt.imshow(img1_background, 'gray'); plt.title("Image 1 Background")
plt.subplot(222); plt.imshow(img1_foreground[:, :, ::-1]); plt.title("Image 1 Foreground")
plt.subplot(224); plt.imshow(img2_background[:, :, ::-1]); plt.title("Image 2 Background")
plt.subplot(223); plt.imshow(img2_foreground, 'gray'); plt.title("Image 2 Foreground")


# There we have it. What we are interested in now is the two color images i.e.
# Image 1 foreground and Image 2 Background. We can easily combine these images
# using OpenCV's `add()` functionality. The reason this will work is because,
# essentially, the places with color on one image are 0 on the other image
# i.e. if Image_1_foreground(x, y) != 0, Image_2_background(x, y) == 0 and viceversa.
# Hence all we are doing is addition by zero.

# In[36]:


#To demonstrate, multiplying the two images should yield 0 at every point
val, = np.unique(img1_foreground * img2_background)
val


# In[37]:


#Let's add up the images
#We use cv2.add to properly handle overflow
fin_img = cv2.add(img1_foreground, img2_background)
plt.figure(figsize=[3, 3])
plt.imshow(fin_img[:, :, ::-1])


# Voila!!!

# ## Exercise

# You are given two images, generate the following result.
# 
# ![Exercise-03-preview](https://opencv.org/wp-content/uploads/2021/08/c0-m2-Exercise-03-preview.png)

# In[58]:


#load the images
green_circle = cv2.imread('images/green_circle.png', cv2.IMREAD_COLOR)
#generate a yellow square
yellow_square = np.ones_like(green_circle) * [255, 255, 0]

#plotting
plt.figure(figsize=[8, 3])
plt.subplot(121); plt.imshow(green_circle)
plt.subplot(122); plt.imshow(yellow_square)


# In[80]:


#We want to create a black circle from the green_circle
_, green_circ_thresh = cv2.threshold(cv2.cvtColor(green_circle.copy(), \
                                cv2.COLOR_BGR2GRAY), 128, 255, cv2.THRESH_BINARY)
plt.figure(figsize=[3, 3])
plt.imshow(green_circ_thresh, 'gray')


# In[84]:


#Simply, bitwise_and(yellow_square, yellow_square, mask= "the inverse of green_circ_thresh")
inv_thresh = cv2.bitwise_not(green_circ_thresh)
final = cv2.bitwise_and(yellow_square, yellow_square, mask=inv_thresh)
plt.figure(figsize=[3, 3])
plt.imshow(final)


# Voila!!!

# We have now seen how to apply the mask and have gone on to apply it to another example.

# ## Another Exercise: adding An image onto another

# We are going to add two images, i.e. add a small image to a bigger image. Let's load up the images.

# In[19]:


smaller = cv2.imread("images/X.png", cv2.IMREAD_COLOR)
bigger = cv2.imread("images/colorful_wallpaper.jpg", cv2.IMREAD_COLOR)

#showing the images
plt.figure(figsize=[8, 3])
plt.subplot(121); plt.imshow(smaller[:, :, ::-1])
plt.subplot(122); plt.imshow(bigger[:, :, ::-1])


# In[20]:


#We want to place X in the centre of the wallpaper,
#so we take the ROI
cols = bigger.shape[1]
rows = bigger.shape[0]
roi = bigger[int(rows/4):int(rows*0.75), int(cols/4):int(cols*0.75)]
plt.figure(figsize=[3, 3])
plt.imshow(roi[:, :, ::-1])


# We will now place on this `roi` image, the smaller image, the X logo.

# In[24]:


#The bitwise_and operation will help us achieve this
#bitwise_and on the roi returns the roi
#We then and it with the mask which is only true where the mask != 0
#hence we invert the smaller
_, smaller_thresh = cv2.threshold(cv2.cvtColor(smaller.copy(),\
                        cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY_INV)
smaller_roi = cv2.bitwise_and(roi, roi, mask=smaller_thresh)
plt.figure(figsize=[3, 3])
plt.imshow(smaller_roi[:, :, ::-1])


# In[25]:


#Now return the image to the original image
bigger[int(rows/4):int(rows*0.75), int(cols/4):int(cols*0.75)] = smaller_roi
plt.figure(figsize=[5, 5])
plt.imshow(bigger[:, :, ::-1])


# In[ ]:

#Saving the image
cv2.imwrite('embedded_logo.png', bigger)



