#!/usr/bin/env python
# coding: utf-8

# # Histograms and color Segmentation

# ## Histograms

# A histogram is a visual representation of the distribution of quantitative data.
# Data is grouped into bars. The length of the bars represents the frequencies of
# the group, while the width of the bars represents the size of the group (known as bin).
# Essentially, each bar tells us that there are `n` occurrences (reading from
# the vertical axis) of values in the range `x - y` (reading from the horizontal axis).

# We can develop histograms of images to see how the color values are distributed.
# Let us see a few examples.

# In[19]:


#Load images
import numpy as np
import matplotlib.pyplot as plt
import cv2

plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['figure.figsize'] = [20, 5]


# We will first examine a black image.

# In[20]:


bw_img = np.zeros([20, 20])

plt.figure()
plt.subplot(121); plt.imshow(bw_img); plt.title('Black Image');
plt.subplot(122); plt.hist(np.ravel(bw_img), range=[0, 255]);\
    plt.title('Histogram'); plt.xlabel('Bins'); plt.ylabel('frequencies')


# Above, we plot an image and its corresponding histogram. We use the default
# binsize of 10 and tell that the range of values is 0 - 255.
# Hence each bin has a width of 25.5.
# We can interpret the histogram as: there are at most 400
# values in the image, all falling in the range 0 - 25.

# Let us now read a b/w image.

# In[21]:


checker = cv2.imread('images/checker_pattern.png', cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.subplot(131); plt.imshow(checker); plt.title('CheckerBoard')

plt.subplot(132); plt.hist(np.ravel(checker), range=[0, 255]);\
    plt.title('Histogram (10 Bins)'); plt.xlabel('Bins'); plt.ylabel('frequencies')

plt.subplot(133); plt.hist(np.ravel(checker), 50, range=[0, 255]);\
    plt.title('Histogram (50 Bins)'); plt.xlabel('Bins'); plt.ylabel('frequencies')


# We can see that changing the nummber of bins changes the shape of the histograms.
# We see that the values are not pure b/w but rather
# there are smaller transitions to the blacks and whites.

# We now look at another b/w image. We will use both
# numpy's `hist` and OpenCV's `CalcHist` function.

# In[30]:


chaplin = cv2.imread('images/charlie_chaplin_bw.jpg', cv2.IMREAD_GRAYSCALE)

chap_hist = cv2.calcHist([chaplin], [0], None, [256], ranges=[0, 255])

plt.figure()
plt.subplot(131); plt.imshow(chaplin); plt.title('Charlie Chaplin');

plt.subplot(132); plt.hist(np.ravel(chaplin), 256, range=[0, 255]);\
    plt.title('Pyplot Histogram'); plt.xlabel('Bins'); plt.ylabel('frequencies')

plt.subplot(133); plt.plot(chap_hist); plt.title('OpenCV Histogram');\
    plt.xlabel('Bins'); plt.ylabel('frequencies')


# In the plot above, we have used 256 bins, so that each value is in its own range.
# We see that the pics are majorly close to the ends, both left and right.

# ## Color Histograms

# We can also view histograms of color images.

# In[32]:


hibiscus = cv2.imread('images/red_hibiscus.jpg')

#The 3 channels
blue = cv2.calcHist([hibiscus], [0], None, [256], ranges=[0, 256])
green = cv2.calcHist([hibiscus], [1], None, [256], ranges=[0, 256])
red = cv2.calcHist([hibiscus], [2], None, [256], ranges=[0, 256])

plt.figure()
plt.subplot(131); plt.imshow(hibiscus); plt.title('Hibiscus');
plt.subplot(132); plt.plot(blue, 'b');
plt.subplot(132); plt.plot(green, 'g');
plt.subplot(132); plt.plot(red, 'r');


# ### Applying a mask 

# We can apply a mask to our image to only get the histogram from a specific portion.

# In[45]:


mask = np.zeros_like(hibiscus)

mask[840:1050, 320:600] = [255, 255, 255]

hibiscus_mask = cv2.bitwise_and(hibiscus, mask)

#The 3 channels
blue = cv2.calcHist([hibiscus], [0], mask[:, :, 0], [256], ranges=[0, 256])
green = cv2.calcHist([hibiscus], [1], mask[:, :, 0], [256], ranges=[0, 256])
red = cv2.calcHist([hibiscus], [2], mask[:, :, 0], [256], ranges=[0, 256])

plt.figure()
plt.subplot(131); plt.imshow(hibiscus_mask); plt.title('Hibiscus');
plt.subplot(132); plt.plot(blue, 'b'); plt.title('HIstogram');
plt.subplot(132); plt.plot(green, 'g');
plt.subplot(132); plt.plot(red, 'r');


# This has given us the histogram values of the desired petal.

# ## Histogram Equalization for Gray Images

# We can use histogram equalization to increase contrast in an image.
# OpenCV provides the `cv2.equalizeHist` function. We will forst load up a gray image.

# In[49]:


unequalized_bw = cv2.imread('images/overexposed_kids_playing.jpg', cv2.IMREAD_GRAYSCALE)

uneqhist = cv2.calcHist([unequalized_bw], [0], None, [256], ranges=[0, 255])

plt.figure()
plt.subplot(121); plt.imshow(unequalized_bw); plt.title('Unequalized Image')
plt.subplot(122); plt.plot(uneqhist); plt.title('Histogram')


# We can see that the image is overexposed and that the values is not evenly
# distributed. Let us look at it after equalizing the histogram.

# In[51]:


#Equalize histogram
equalized_bw = cv2.equalizeHist(unequalized_bw)

equalhist = cv2.calcHist([equalized_bw], [0], None, [256], ranges=[0, 255])

plt.figure()
plt.subplot(121); plt.imshow(equalized_bw); plt.title('Equalized Image')
plt.subplot(122); plt.plot(equalhist); plt.title('Histogram')


# We can now see that the colors are now better distributed than in the original image.

# ## Histogram Equalization for Color Images

# Equalizing color images is not as straightfoward as gray images. We cannot simply
# equalize the channels and output the resulting image.
# This naive approach is shown below.

# In[54]:


#The naive approach
unequalized_color = cv2.imread('images/dim_color_unequalized_hist.jpg')

naive_equalize = unequalized_color.copy()

for x in range(3):
    naive_equalize[:, :, x] = cv2.equalizeHist(naive_equalize[:, :, x])

plt.figure(figsize=[10, 5])
plt.subplot(121); plt.imshow(unequalized_color[:, :, ::-1]);\
    plt.title('Original color Image')

plt.subplot(122); plt.imshow(naive_equalize[:, :, ::-1]);\
    plt.title('Naively Equalized Image')


# From the equalized image, we see that artifacts have been introduced.
# We can see that colors that don't exist have been introduced. A better way
# is to convert the image into HSV color space and equalizing the Value channel.

# In[55]:


#convert to hsv
unequalized_hsv = cv2.cvtColor(unequalized_color, cv2.COLOR_BGR2HSV)

#equalize the value channel
unequalized_hsv[:, :, 2] = cv2.equalizeHist(unequalized_hsv[:, :, 2])

pro_equalized = cv2.cvtColor(unequalized_hsv, cv2.COLOR_HSV2BGR)

plt.figure(figsize=[10, 5])
plt.subplot(121); plt.imshow(unequalized_color[:, :, ::-1]);\
    plt.title('Original color Image')

plt.subplot(122); plt.imshow(pro_equalized[:, :, ::-1]);\
    plt.title('Pro Equalized Image')


# We see that the resultant image is better than the one we had previously.
# Let us plot the histograms to see.

# In[67]:


#plotting histograms

#Original image
blue_org = cv2.calcHist([unequalized_color], [0], None, [256], ranges=[0, 255])
green_org = cv2.calcHist([unequalized_color], [1], None, [256], ranges=[0, 255])
red_org = cv2.calcHist([unequalized_color], [2], None, [256], ranges=[0, 255])

#Naively equalized color
blue_naive = cv2.calcHist([naive_equalize], [0], None, [256], ranges=[0, 255])
green_naive = cv2.calcHist([naive_equalize], [1], None, [256], ranges=[0, 255])
red_naive = cv2.calcHist([naive_equalize], [2], None, [256], ranges=[0, 255])

#Pro equalized color
blue_pro = cv2.calcHist([pro_equalized], [0], None, [256], ranges=[0, 255])
green_pro = cv2.calcHist([pro_equalized], [1], None, [256], ranges=[0, 255])
red_pro = cv2.calcHist([pro_equalized], [2], None, [256], ranges=[0, 255])

#Plotting
plt.figure(figsize=[20, 5])

#Original image
plt.subplot(131); plt.plot(blue_org, 'b'); plt.title('Original Color Image');
plt.subplot(131); plt.plot(green_org, 'g');
plt.subplot(131); plt.plot(red_org, 'r');

#Naively equalized color
plt.subplot(132); plt.plot(blue_naive, 'b'); plt.title('Naively Equalized Image');
plt.subplot(132); plt.plot(green_naive, 'g'); 
plt.subplot(132); plt.plot(red_naive, 'r');

#Pro equalized color
plt.subplot(133); plt.plot(blue_pro, 'b'); plt.title('Pro Equalized Image');
plt.subplot(133); plt.plot(green_pro, 'g'); 
plt.subplot(133); plt.plot(red_pro, 'r');


# We see that there are differences between the two histograms. Whereas the naively
# equalized image seems more equalized, the pro equalized image gives better results.
