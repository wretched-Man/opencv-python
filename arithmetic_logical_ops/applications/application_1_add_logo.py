#!/usr/bin/env python
# coding: utf-8

# # Applications of bitwise Operations

# ## Watermarking

# We are going to add a watermark logo to an image. Let's load up the images.

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = [10, 5]


# In[2]:


watermark = cv2.imread('python-logo-only.png', cv2.IMREAD_UNCHANGED)
img = cv2.imread('adventure_nowatermark.jpg', cv2.IMREAD_COLOR)

watermark_show = cv2.cvtColor(watermark, cv2.COLOR_BGRA2RGBA)

plt.figure()
plt.subplot(121); plt.imshow(img[:, :, ::-1]); plt.title('Image to Watermark')
plt.subplot(122); plt.imshow(watermark_show);  plt.title('Watermark')


# The image is okay. We may need to resize it, however.

# In[3]:


r_watermark = cv2.resize(watermark, None, fx=2, fy=2, interpolation=cv2.INTER_LINEAR)
r_watermark_show = cv2.cvtColor(r_watermark, cv2.COLOR_BGRA2RGBA)
plt.figure()
plt.imshow(r_watermark_show)


# In[4]:


r_watermark.shape


# Now we can begin.

# In[5]:


#First, we take a ROI, centre
logo_r = r_watermark.shape[0]
logo_c = r_watermark.shape[1]
row0 = int((img.shape[0] - logo_r)/2)
col0 = int((img.shape[1] - logo_c)/2)
roi = img[row0:row0 + logo_r, col0:col0 + logo_c]

plt.figure()
plt.imshow(roi[:, :, ::-1])


# In[6]:


#Next, we want to make the ROI the background of our image
#We can use the alpha channel in case of a 4-channel image 

#We first split the bgr and alpha portions of the logo
logo_bgr = r_watermark[:, :, :3] #BGR
logo_alpha = r_watermark[:, :, 3] #alpha

#We then replicate the alpha channel to fit 3 channels, this
#so as to use in the bitwise_and operation
alpha_3channel = cv2.merge([logo_alpha, logo_alpha, logo_alpha])

#We mask the ROI image with inverse(logo_alpha)
masked_roi = cv2.bitwise_and(roi, roi, mask=cv2.bitwise_not(logo_alpha))

#We then combine the BGR image with the alpha image to
#isolate the logo portion. The outer region is value 0
combined_alpha_bgr = cv2.bitwise_and(logo_bgr, alpha_3channel)

#We then add/ or the masked ROI image with the
#combined_alpha_bgr to have our watermark
roi_watermark = cv2.add(masked_roi, combined_alpha_bgr)

plt.figure(figsize=[20, 20])
plt.subplot(331); plt.imshow(logo_bgr[:, :, ::-1]); plt.title("Logo BGR")
plt.subplot(332); plt.imshow(logo_alpha); plt.title("Logo Alpha")
plt.subplot(333); plt.imshow(alpha_3channel[:, :, ::-1]); plt.title("3-Channel Alpha")
plt.subplot(334); plt.imshow(masked_roi[:, :, ::-1]); plt.title("Masked ROI")
plt.subplot(335); plt.imshow(combined_alpha_bgr[:, :, ::-1]); plt.title("Combined Alpha BGR")
plt.subplot(336); plt.imshow(roi_watermark[:, :, ::-1]); plt.title("ROI Watermark")


# Whereas the logo is black on white, it stands out of the background as white,
# hence we made it so. Otherwise, `roi_temp

# In[7]:


#Now we return the roi to the original image
img_1 = img.copy()
img_1[row0:row0 + logo_r, col0:col0 + logo_c] = roi_watermark
plt.figure()
plt.imshow(img_1[:, :, ::-1])


# There we have it, our watermark right in the middle. Although it is too visible.
# We want to make it less bright. This, we can simply do by changing the `cv2.add`
# function and using `cv2.addWeighted`.

# For any point `P(x, y)`, the value of the destination pixel is calculated as:
# ``` python
# dst = src1*alpha + src2*beta + gamma
# ```

# In[8]:


#[row0:row0 + 730, col0:col0 + 730]
roi_2 = roi.copy()
roi_light_watermark = cv2.addWeighted(roi_2, 1, combined_alpha_bgr, 0.5, 0)
plt.figure()
plt.imshow(roi_light_watermark[:, :, ::-1])


# In[9]:


img_2 = img.copy()
img_2[row0:row0 + logo_r, col0:col0 + logo_c] = roi_light_watermark
plt.figure()
plt.imshow(img_2[:, :, ::-1])


# In[13]:


cv2.imwrite('adventure_light_python_watermark.jpg', img_2)
cv2.imwrite('adventure_solid_python_watermark.jpg', img_1)




