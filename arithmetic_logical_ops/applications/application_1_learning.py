#!/usr/bin/env python
# coding: utf-8

# # Testing a light watermark on an image

# Given an ROI and the same ROI with a light watermark (one added using `cv2.addWeighted`),
# what differentiates the two images?

# In[44]:


#Testing what a light watermark does to an image
#loading up the image
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = [10, 5]


# In[2]:


roi = cv2.imread('images/adventure_roi_nowatermark.jpg', cv2.IMREAD_COLOR)
w_roi = cv2.imread('images/roi_light_python_watermark.jpg', cv2.IMREAD_COLOR)

print(roi.shape)
print(w_roi.shape)


# In[3]:


plt.figure()
plt.subplot(121); plt.imshow(roi[:, :, ::-1]); plt.title("ROI")
plt.subplot(122); plt.imshow(w_roi[:, :, ::-1]); plt.title("Watermarked ROI")


# What is curious with this question is the fact that the watermark is transparent.
# It looks as if it is floating? Is it only a lighter color or what is it?

# In[4]:


plt.figure()
plt.subplot(121); plt.imshow(roi[:, :, 1]); plt.title('ROI Green')
plt.subplot(122); plt.imshow(w_roi[:, :, 1]); plt.title('WaterMarked ROI Green')


# We will examine the green channel of both images now. In it, most of the logo
# watermark is bright in most portions. We will first begin with comparing in
# numpy the values that are not equal.

# In[5]:


#isolate the greens
w_green = w_roi[:, :, 1]
roi_green = roi[:, :, 1]

#build a boolean array of unequal coordinates
unequal_total = (w_green != roi_green)

#Use the boolean array to isolate those values in
#w_green and roi_green
w_unequal = w_green[unequal_total]
roi_unequal = roi_green[unequal_total]

#Should print True... test condition
print('All unequal?', (w_unequal != roi_unequal).all())

#Print percentage of unequal values
print('Percentage unequal:', 100 * np.prod(w_unequal.shape)/np.prod(w_green.shape), '%')


# We see that 63% of the values are unequal. For the true test,
# we want to see where this inequality lies. Is one greater, and another less?
# For this we can plot the values and see.

# In[6]:


plt.figure(figsize=[20, 7])
plt.subplot(211); plt.plot(roi_unequal, color='green'); plt.title('ROI Green Values')
plt.subplot(212); plt.plot(w_unequal, color='blue'); plt.title('Watermarked ROI Green Values')


# What we see from the above graphs is that for the Watermarked ROI,
# the values are higher. Let us see the 3D plots.

# In[54]:


#Build mosaic
mosaic = [['Green ROI Top',            'Green ROI Side'],
         ['Watermarked ROI Top', 'Watermarked ROI Side']]

#Build X, Y coordinates
X = np.arange(538)
Y = np.arange(652)
X, Y = np.meshgrid(X, Y)
fig, axs = plt.subplot_mosaic(mosaic, subplot_kw={"projection": "3d"})


#Green ROI top
axs['Green ROI Top'].plot_surface(X, Y, roi_green, vmin=0, cmap=matplotlib.cm.Greens)
axs['Green ROI Top'].elev = 90
axs['Green ROI Top'].azim = 90

#Green ROI Side
axs['Green ROI Side'].plot_surface(X, Y, roi_green, vmin=0, cmap=matplotlib.cm.Greens)
axs['Green ROI Side'].elev = 0
axs['Green ROI Side'].azim = 180

#Watermarked ROI Top
axs['Watermarked ROI Top'].plot_surface(X, Y, w_green, vmin=0, cmap=matplotlib.cm.Greens)
axs['Watermarked ROI Top'].elev = 90
axs['Watermarked ROI Top'].azim = 90

#Watermarked ROI Side
axs['Watermarked ROI Side'].plot_surface(X, Y, w_green, vmin=0, cmap=matplotlib.cm.Greens)
axs['Watermarked ROI Side'].elev = 0
axs['Watermarked ROI Side'].azim = 180


# Here, we see that the only difference between these two is the pixel values only.





