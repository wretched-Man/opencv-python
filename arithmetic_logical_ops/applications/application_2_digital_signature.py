#!/usr/bin/env python
# coding: utf-8

# # Application 2 - Digital signature

# For the second application of bitwise operations, we are going to
# create a digital signature from a user's signature image.

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [10, 5]
plt.rcParams['image.cmap'] = 'gray'


# In[2]:


#lets load up the images
rawsig = cv2.imread('images/application_2_raw_signature.jpg', cv2.IMREAD_COLOR)

plt.figure()
plt.imshow(rawsig[:, :, ::-1])


# In[3]:


#We crop the signature
croppedsig = rawsig.copy()[500:1400, 1250:2700]

plt.figure()
plt.imshow(croppedsig[:, :, ::-1])
croppedsig.shape


# The signature is of dimension `1450 x 900`.
# We may want to keep it this way for quality purposes.

# In[4]:


#Next, we create a mask for the image
_, croppedmask = cv2.threshold(cv2.cvtColor(croppedsig,\
                                cv2.COLOR_RGB2GRAY), 110, 255, cv2.THRESH_BINARY)

plt.figure()
plt.imshow(croppedmask)


# In[5]:


#using erosion and dilaiton to remove some artifacts

kernel = np.ones((4, 4), np.uint8)
croppeddilate = cv2.dilate(croppedmask, kernel, iterations=1)

croppederode = cv2.erode(croppeddilate, kernel, iterations=1)

plt.figure()
plt.imshow(croppederode)


# In[12]:


#create a blue mask
bluemask = np.ones(croppedsig.shape, dtype=np.uint8) * np.uint8((255, 0, 0))

plt.figure()
plt.imshow(bluemask[:, :, ::-1])


# In[15]:


#Since bluemask is 3-channel, we now make croppederode so
bwsig = cv2.merge([croppederode, croppederode, croppederode])

#We then add. Since a darker blue gives it a more realistic feel,
#we use addWeighted.
cleansig = cv2.addWeighted(bwsig, 1, bluemask, 0.6, 0)

plt.figure()
plt.imshow(cleansig[:, :, ::-1])


# We now have our signature. Let's save it.

# In[16]:


cv2.imwrite('application_2_final_signature.jpg', cleansig)


# ## Making our signature transparent.

# We can do more to our signature, we can make it transparent. This, we can do simply.

# In[23]:


cleansig_alpha = cv2.merge([cleansig, cv2.bitwise_not(croppederode)])

cv2.imwrite('application_2_final_alpha_signature.png', cleansig_alpha)


# We read the image again to see it.

# In[24]:


from IPython.display import Image

Image(filename='application_2_final_alpha_signature.png', width=300)


# There we have it, our transparent signature.





