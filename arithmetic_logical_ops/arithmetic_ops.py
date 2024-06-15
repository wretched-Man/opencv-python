#!/usr/bin/env python
# coding: utf-8

# # Arithmetic Operations

# OpenCV allows us to perform arithmetic operations on images. It provides methods like:
# * `add()`
# * `subtract()`
# * `multiply()`

# In[3]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# Addition and Subtraction

# In[7]:


img = cv2.imread('images/mountains.jpg', cv2.IMREAD_COLOR)
plt.figure(figsize=[5, 5])
plt.imshow(img[:, :, ::-1])


# In[21]:


#Let's add using OpenCV
matrix = np.ones(img.shape, np.uint8) * 45
(matrix == 45).all()


# In[25]:


img_add = cv2.add(img, matrix)
plt.imshow(img_add[:, :, ::-1])


# In[26]:


#Let's now subtract
img_sub = cv2.subtract(img, matrix)
plt.imshow(img_sub[:, :, ::-1])


# Let us plot all together.

# In[30]:


plt.figure(figsize=[20, 5])
plt.subplot(131); plt.imshow(img_sub[:, :, ::-1]); plt.title('Subtracted (Darker)')
plt.subplot(132); plt.imshow(img[:, :, ::-1]);     plt.title('Original')
plt.subplot(133); plt.imshow(img_add[:, :, ::-1]); plt.title('Added (Brighter)')


# As you can see, addition can be used to make an image brighter.
# Subtraction can be used to make an image darker.

# ### An NB about addition & subtraction

# To illustrate this, we are going to use a matrix of value `100`.

# In[31]:


#Building the matrix
matrix_2 = np.ones(img.shape, np.uint8) * 100


# In[32]:


cv_add = cv2.add(img, matrix_2)

#Since the images are np arrays, we can just add using numpy
np_add = img + matrix_2


# In[40]:


#plotting
plt.figure(figsize=[13, 5])
plt.subplot(131); plt.imshow(cv_add[:, :, ::-1]); plt.title('OpenCV Addition')
plt.subplot(133); plt.imshow(np_add[:, :, ::-1]); plt.title('NumPy Addition')
plt.subplot(132); plt.imshow(img[:, :, ::-1]); plt.title('Original')


# Since OpenCV `Mat` objects are numpy arrays, we can also use numpy addition to add the images.
# However, when we do that we see that the operations yield very different results for OpenCV
# and for NumPy. Compared to the original, the OpenCV added image appears brighter, as expected, 
# whereas the Numpy added image contains artifacts.
# This is because the way Numpy and OpenCV add differs significantly, as shown below.

# In[51]:


#A little examination
print('cv_add', len(cv_add[cv_add == 255]))
print('np_add', len(np_add[np_add == 255]))


# The above results may not tell us much, only that cv_add has more `255` than np_add.
# How does that help us, you ask? It's significant since it shows us how NumPy and
# OpenCV differ in addition. OpenCV clips overflow values (values greater than 255) at 255,
# whereas NumPy performs addition modulo `%` 256 (that is, it takes the remainder of dividing a number by 256).
# 
# For example:
# # For NumPy, `100 + 236 = 336 % 256 = 80`
# For OpenCV, `100 + 236 = 336 = 255`
#
# Hence the reason why we have more `255`'s in the OpenCV added image than in the NumPy one.
# The NumPy one has artifacts which develop from the modulo operation.

# We can prevent this behavior in NumPy by using `np.clip()`, and specify the values from `0` to `255`.

# In[59]:


np_add2 = np.uint8(np.clip(np.add(img, matrix_2, dtype=np.int32), 0, 255))


# Instead of using the `+` operator, we have used `np.add`.
# Although they serve the same functionality, `np.add` allows more operations.
# For example, here we specify that we want the result to be of type `int32`.
# We then cast the clipped image `uint8`.

# In[62]:


#We can easily check even without plotting
(np_add2 == cv_add).all()


# ### What of Subtraction?

# In[114]:


cv_sub = cv2.subtract(img, matrix_2)
np_sub = img - matrix_2


# In[115]:


#plotting
plt.figure(figsize=[13, 5])
plt.subplot(131); plt.imshow(cv_sub[:, :, ::-1]); plt.title('OpenCV Subtraction')
plt.subplot(133); plt.imshow(np_sub[:, :, ::-1]); plt.title('NumPy Subtraction')
plt.subplot(132); plt.imshow(img[:, :, ::-1]); plt.title('Original')


# We see that, as expected, the OpenCV image gets darker whereas the NumPy image has artifacts.
# Let us examine the image.

# In[118]:


#Let us observe a single point [0, 0]
print('Original', img[0, 0])
print('np_sub  ', np_sub[0, 0])
print('cv_sub  ', cv_sub[0, 0])


# We observe for the first point they are similar.
#
# `B` channel
# `120 - 100 = 20` for both of them.
#
# `G` channel
# OpenCV Subtraction
# `45 - 100 = -55 = 0`
# Numpy Subtraction
# # `45 - 100 = -55 + 256 = 201 `
# 
# `R` channel
# OpenCV Subtraction
# `19 - 100 = -81 = 0`
# Numpy Subtraction
# `19 - 100 = -81 + 256 = 175 `
#
#So, we see that for numpy, the values wrap around since we are using `uint8` (0 - 255).
# There's also another way to see this.
# Since the matrix is full of `100`, the largest negative value we expect is `-100`
# which wraps to `156`. Hence, where `np_sub` and `cv-sub` deviate, we expect to see 
# the values range from `156 - 255`. At the same time, at those indices, `cv_sub` should
# have a value of `0` since OpenCV clips values not in the range (0, 255). Let's test this out.

# In[135]:


#We first find the places where they deviate...
deviate_arr = np.array((np_sub != cv_sub).flat)
#True values indicate that the arrays deviate
deviate_arr


# In[132]:


#We expect that, in np_sub, the values of the indices that deviate range from [156 - 255]
print(np_sub.flat[deviate_arr].min(), '-', np_sub.flat[deviate_arr].max())


# In[134]:


#We also expect that, in cv_sub, the values of the indices that deviate are 0
uniqv, = np.unique(cv_sub.flat[deviate_arr])
#We create an array of unique(non-repeating) values in the deviating indices of cv_sub,
#we expect only one unique value, 0
uniqv


# As we expected, so have we seen! 

# Multiplication

# We can also multiply images in OpenCV using `cv2.multiply()`.

# In[137]:


bright = cv2.imread('images/outdoor_colorful.jpg', cv2.IMREAD_COLOR)

plt.figure(figsize=[5, 5])
plt.imshow(bright[:, :, ::-1])


# In[138]:


mult_ax = np.ones(bright.shape, np.float64) * 0.8
mult_ax2 = np.ones(bright.shape, np.float64) * 1.2


# In[143]:


#We pass bright as float64 to cv2.multiply
#We cast the result to uint8
low_bright = np.uint8(cv2.multiply(np.float64(bright), mult_ax))
high_bright = np.uint8(cv2.multiply(np.float64(bright), mult_ax2))


# In[144]:


plt.figure(figsize=[13, 5])
plt.subplot(131); plt.imshow(low_bright[:, :, ::-1]); plt.title('Low contrast')
plt.subplot(133); plt.imshow(high_bright[:, :, ::-1]); plt.title('High contrast')
plt.subplot(132); plt.imshow(bright[:, :, ::-1]); plt.title('Original')


# One thing to note is that we cast the result to `np.uint8` and the reason for this is
# that Matplotlib's `imshow` function expects the range `0, 1` for floats,
# hence it truncates the values.

# As we can see, multiplication is concerned with changing the contrast of the image.
# In the high contrast image, we see artifacts. Let us examine those.

# In[147]:


#Let us multiply using numpy to see if we get the same image.
np_high_bright = np.uint8(bright * mult_ax2)
#Let's compare
(np_high_bright == high_bright).all()


# The images are similar, showing us that for multiplication, unlike addition and subtraction,
# OpenCV behaves just like NumPy and does not clip the values but performs modulo operation on them.
# We can prevent this, just like we did for addition, using NumPy's `np.clip()`.

# In[148]:


cv_high_bright2 = np.uint8(np.clip(cv2.multiply(np.float64(bright), mult_ax2), 0, 255))


# In[151]:


#Plotting
plt.figure(figsize=[13, 5])
plt.subplot(131); plt.imshow(low_bright[:, :, ::-1]); plt.title('Low contrast')
plt.subplot(133); plt.imshow(cv_high_bright2[:, :, ::-1]); plt.title('High contrast')
plt.subplot(132); plt.imshow(bright[:, :, ::-1]); plt.title('Original')


# We now have a high contrast image - although we have also lost some detail. We can also achieve this with NumPy.

# In[155]:


np_high_bright2 = np.uint8(np.clip(np.multiply(bright, mult_ax2), 0, 255))
#Compare
(np_high_bright2 == cv_high_bright2).all()


# To recap, OpenCV supports arithmetic operations such as addition, subtraction and multiplication.
# For addition and subtraction, these operations differ significantly with NumPy, since OpenCV clips
# the values to the range `[0, 255]`, whereas NumPy wraps around the value (assuming `np.uint8`).
# 
# For multiplication, NumPy and OpenCV behave similarly by wrapping around excess values.
# We can change this behavior and perform clipping by using `numpy.clip()` method.
