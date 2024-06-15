#!/usr/bin/env python
# coding: utf-8

# # The Alpha Channel

# This is the fourth channel and exists in formats such as PNG.
# It has a range of **0 - 255** with **0** signifying full transparency.

# ## Examine Alpha Channel

# We will examine the Alpha Channel of a logo.

# In[103]:


import cv2
import matplotlib.pyplot as plt
import numpy as np
from IPython.display import Image
plt.rcParams['image.cmap'] = 'gray' #colormap for gray images


# In[104]:


fblogo = cv2.imread('images/Facebook_Logo_Primary.png', cv2.IMREAD_UNCHANGED)
print(fblogo.shape)


# In[114]:


#Split the image to requisite channels
b, g, r, a = cv2.split(fblogo)

plt.figure(figsize=[14, 5])
plt.subplot(141); plt.imshow(b, vmin=0, vmax=255); plt.title('Blue Channel')
plt.subplot(142); plt.imshow(g, vmin=0, vmax=255); plt.title('Green Channel')
plt.subplot(143); plt.imshow(r, vmin=0, vmax=255); plt.title('Red Channel')
plt.subplot(144); plt.imshow(a, vmin=0, vmax=255); plt.title('Alpha Channel')


# In[115]:


print('Blue', np.unique(fblogo[:, :, 0]))
print('Green', np.unique(fblogo[:, :, 1]))
print('Red', np.unique(fblogo[:, :, 2]))
print('Alpha', np.unique(fblogo[:, :, 3]))


# A few observations:
# 1. We can see that all the blue channel is saturated (fully white).
# 2. We can see that, outside the circle, the green and red are also saturated (Also white).
# 3. The Alpha channel is white within the circle and black outside the circle -
#    this means that allows transparency only within the circle.
# 4. There are different levels of transparency in the alpha channel.
#    We see this from the fact that it has values ranging from `0 - 255` (in the code above.)

# We note that although the other channels have intense values outside the circle,
# the alpha channel ensures that the only colors that can be seen are those\
# within the circle as the ones outside made transparent.

# ## Adding an Alpha channel to an image

# We want to add an Alpha channel to an image. In order to do this,
# we need to specify what parts of the image we want to hide and those we want to see.

# In[117]:


#Let's load up the image
comp_noalpha = cv2.imread("images/company_noalpha.jpg", cv2.IMREAD_UNCHANGED)
print(comp_noalpha.shape)

plt.figure(figsize=[3, 3])
plt.imshow(comp_noalpha[:, :, ::-1])


# We see that the unchanged image has 3 channels. We aim to make the background transparent.

# In[122]:


#First, we make a mask
_, noalpha_thresh = cv2.threshold(cv2.cvtColor(comp_noalpha,\
                                    cv2.COLOR_BGR2GRAY), 127, 255, cv2.THRESH_BINARY)

plt.figure(figsize=[3, 3])
plt.imshow(noalpha_thresh, vmin=0, vmax=255)


# If you can believe it, we now have our mask. The dark areas will be completely
# hidden and the white areas completely revealed. Let's now add the channel.

# In[127]:


#adding the alpha channel
comp_withalpha = np.ones((comp_noalpha.shape[0],\
                          comp_noalpha.shape[1], 4), dtype = comp_noalpha.dtype)
comp_withalpha[:, :, 0:3] = comp_noalpha
comp_withalpha[:, :, 3] = noalpha_thresh #adding the alpha channel

print(comp_withalpha.shape)


# In[128]:


#In order to see the image, we use Ipython since it is a BGR image... so we save it first
cv2.imwrite("company_withalpha.png", comp_withalpha) #We save as png which supports alpha channel

#Loading up the image
Image(filename="company_withalpha.png", width=300)


# Voila! Our image is now transparent. Because of the alpha channel

# ### Comparing with vs without alpha

# In[131]:


#To plot with matplotlib, we convert to RGBA
comp_withalpha = cv2.cvtColor(comp_withalpha.copy(), cv2.COLOR_BGRA2RGBA)

plt.figure(figsize=[10, 4])
plt.subplot(121); plt.imshow(comp_noalpha[:, :, ::-1]); plt.title('Without Alpha'); plt.axis('off')
plt.subplot(122); plt.imshow(comp_withalpha); plt.title('With Alpha'); plt.axis('off')


# The with_alpha image is transparent and blends into any background
# whereas the without_alpha image black and will only 'blend' with a black background.
