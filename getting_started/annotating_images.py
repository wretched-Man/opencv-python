#!/usr/bin/env python
# coding: utf-8

# ## Annotating Images

# A few things to note:
# - When annotating images, read them in as color images, even if they are gray. This allows you to annotate the image with color.
# - OpenCV offers built-in functions for drawing (`rectangle()`, `circle()`, `line()`, polygons, `polylines()` and `putText()`).
# - OpenCV uses `(x, y)` co-ordinate system unlike NumPy's `(y, x)`.

# We are going to load up a grayscale image as color and examine it.

# In[1]:


import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib


# In[2]:


#Reading a gray image as color
apple = cv2.imread("images/applegray.jpeg", cv2.IMREAD_COLOR)

#Reading the same gray image unchanged
un_apple = cv2.imread("images/applegray.jpeg", cv2.IMREAD_UNCHANGED)

print(apple.shape)
print(un_apple.shape)


# In[3]:


fig, axs = plt.subplots(1, 2, figsize=[20, 5])
axs[0].imshow(apple, cmap='gray')
axs[1].imshow(un_apple, cmap="gray")
axs[0].set(title = "3-channel gray")
axs[1].set(title = "Original")


# The images look similar. This is because at any point in the original, say `[30, 56]` there is a value `123`, the 3-channel gray will have the same value across its 3 channels. Hence, any single channel in the 3-channel gray is exactly similar to the Original. Let us examine this.

# # Examining the images
#This portion is a detour and runs from ln[4] - ln[62] ('Back to annotating)

# In[4]:


#Examine the values
for y in range(30, 1200, 43):
    for x in range(30, 1750, 37):
        print(f'({y},{x})', un_apple[y, x], end = ' == ')
        print(f'({y},{x})', apple[y, x])


# From the above, we see that where `un_apple` is a certain number, `n`, `apple` is `[n, n, n]`. It follows, then, that any channel `c` of `apple` is equal to `un_apple`.

# In[5]:


#Showing that any channel c of `apple` is equal to un_apple
for x in range(3):
    print( (apple[:, :, x] == un_apple).all())


# We see that indeed this is true. In essence the ranges of both `apple` and `un_apple` are the same, a maximum of 256 colors. Let us map `apple`'s range to see its colors

# In[6]:


#Building the range
rang = np.arange(256)
#We will create an image with the values [0 - 255] repeated in a 3-channel image to see what colors they are in RGB
colors = np.empty((256, 1024, 3), dtype = np.uint8)

for x in range(0, colors.shape[1], 4):
    colors[:, x:x + 4] = x/4


# In[7]:


plt.figure(figsize=[12, 12])
plt.imshow(colors)


# This is the output color image. Every `[n, n, n]` for 0 <= n < 256 produces a gray image as we would get if we plotted a single channel.

# ### Curiously

# ### How would `apple` look like in HSV?

# In[8]:


apple_hsv = cv2.cvtColor(apple, cv2.COLOR_RGB2HSV)


# In[9]:


apple_hsv.shape


# In[10]:


fig2, ax2 = plt.subplots(1, 3, figsize=[20, 10])
h, s, v = cv2.split(apple_hsv)

#hue
ax2[0].imshow(h, cmap="gray")
ax2[0].set(title="hue")

#saturation
ax2[1].imshow(s, cmap="gray")
ax2[1].set(title="saturation")

#value
ax2[2].imshow(v, cmap="gray")
_= ax2[2].set(title='value')


# It seems that hue(type of color) and saturation(strength/intensity of that color) are at 0.

# In[11]:


print('hue:', np.unique(h))
print('saturation:', np.unique(s))
print('value:', np.unique(v))


# Yes, only value exists, for our image.

# # Back to Annotating

# The other forms of annotation are pretty basic. We are going
# to draw a polygon around the apple.

# In[62]:


#defining the points
#numpy uses (x, y)

pts = np.array([
                [395, 1045],
                [600, 915],
                [800, 930],
                [1025, 1125],
                [1040, 1260],
                [1000, 1520],
                [800, 1665],
                [600, 1690],
                [415, 1600],
                [375, 1345]
               ], np.int32)


#params
isClosed = True #Whether to close the polygon
color = (0, 0, 255) #Specify color in BGR -> Red
thickness = 15 #thickness of the line
#type of line AA(Anti-aliasing) gives smooth lines,
#can also use LINE_8 (jagged lines)

linetype = cv2.LINE_AA


# In[63]:


#OpenCv takes (x, y) coordinates
#We switch y, x -> x, y;
for x in range(pts.shape[0]):
    pts[x] = pts[x][::-1]


# In[64]:


print(pts)


# In[65]:


pts = pts.reshape((-1, 1, 2)) #Must be a n by 2 array


# In[66]:


drawn_apple = cv2.polylines(apple.copy(), [pts], True, color, thickness=thickness, lineType=linetype)

#plot
plt.figure(figsize=[10, 10])
plt.imshow(drawn_apple[:, :, ::-1])


# In order to close the polygon, ensure to pass the points as a list, i.e `[pts]` and not just as an ndarray `pts`.

# In[ ]:

#Save the apple
cv2.imwrite('drawn_gray_apple.png', drawn_apple)


