#!/usr/bin/env python
# coding: utf-8

# # Deforestation Analysis using Color Segmentation

# We have been looking at histograms, histogram equalization,
# color segmentation using HSV color space. We are now going
# to look at one application in the area of deforestation management.

# We will also have a look at using histograms to determine the
# lower and upper bounds for segmentation using in-range function.

# We will have a look at four images showing human settlement
# and deforestation. We will analyze them in HSV and BGR and
# map out the forest cover as a percentage of total land area.

# In[1]:


#loading the images
import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'


# We will now load and display the 1985 and 2011 images.

# In[2]:


image1985 = cv2.imread('images/1985.jpg')
image2011 = cv2.imread('images/2011.jpg')


# In[3]:


plt.figure(figsize=[10, 8])
plt.subplot(121); plt.imshow(image1985[:, :, ::-1]); plt.title('1985')
plt.subplot(122); plt.imshow(image2011[:, :, ::-1]); plt.title('2011')


# We see a comparable difference between the two images.
# What we now want to do is to segment the images. We have
# previously done this in HSV but we can also segment
# according to RGB. We know that vegetation is green,
# hence we will want to segment according to the green channel.
# We will go ahead and plot the histograms for the channels
# for both images to find the correct threshold values.

# ## Color Segmentation in RGB

# In[4]:


#We create a utility function to plot the histograms
#It takes a BGR image and plots the histograms for the
#various channels
def build_hist_rgb(img, title = '', yscale = 'linear'):
    assert(img.shape[2] == 3) #has 3 channels

    #Build the histograms
    blue_hist=cv2.calcHist([img], [0], None, [256], [0, 255])
    green_hist=cv2.calcHist([img], [1], None, [256], [0, 255])
    red_hist=cv2.calcHist([img], [2], None, [256], [0, 255])
    
    fig = plt.figure(figsize=[20, 5])
    fig.suptitle(title)
    #Blue channel
    ax = fig.add_subplot(1, 3, 1)
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)
    ax.plot(blue_hist, color = 'b', label = 'Blue')
    ax.grid()
    ax.legend()
    
    #Green channel
    ax = fig.add_subplot(1, 3, 2)
    
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)
    
    ax.plot(green_hist, color = 'g', label = 'Green')
    ax.grid()
    ax.legend()

    #Red channel
    ax = fig.add_subplot(1, 3, 3)
    
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)

    ax.plot(red_hist, color = 'r', label = 'Red')
    ax.grid()
    ax.legend()

build_hist_rgb(image1985, '1985 - Linear')
build_hist_rgb(image2011, '2011 - Linear')


# Above, we have plotted the histograms for all the channels of
# the 1985 and 2011 image. We can note that the images are
# relatively dark as most pixels are within the range of less
# than 100 for all the channels (the histograms are skewed left).
# We also see that the peak is at between 50 -100. We can get a
# better look at the distribution by using a log y-scale to
# reduce skewness in the data.

# In[5]:


build_hist_rgb(image1985, '1985 - Log', 'log')
build_hist_rgb(image2011, '2011 - Log', 'log')


# In the previous plot, the high peaks overshadowed the rest of
# the regions. Now, however we can properly see that there are
# other values beyond the peaks. We see two peaks, one between
# 70 - 100 and another bump starting from 150 and declining past
# 200 for the 2011 green histogram. For the 1985 green histogram,
# we see only one dominant peak and the bump is smaller. This
# could suggest the two dominant areas in the images, the vegetation
# being the high peak and the small but increasing bump being
# the human settlement. We can use these values to create an
# inRange function. We need only use the green portion and limit
# the values to between 50 and 100 and we will properly segment
# our image. In this example we have used two histograms to set
# the threshold value for all images, for more fine grained access,
# we could use each image's histogram.

# In[9]:


#Build utility function to produce image masks in rgb
def build_masks(img, lower=[0, 50, 0], upper=[255, 100, 255]):
    #For the blue and red channels, we put lower and upper as
    #the utmost min and max values. For the green channel,
    #we confine our value range to 65 - 120

    return cv2.inRange(img, np.array(lower), np.array(upper))


# In[17]:


#We will calculate forest percentage
def percent_forest(img):
    total = img.shape[0] * img.shape[1]

    cover = cv2.countNonZero(img)

    return round((cover / total) * 100, 2)


# In[20]:


#We will now display and build masks for all the images and
#display them side by side We will also display the forest
#cover as a percentage of the whole mass
#capture all images
images = glob.glob('images/*.jpg')
fig = plt.figure(figsize=[10, 15])
count = 1

for image in images:
    #read the image
    img_x = cv2.imread(image)

    #build a mask
    img_mask = build_masks(img_x)

    #display image and mask
    ax = fig.add_subplot(4, 2, count)
    ax.imshow(img_x[:, :, ::-1])
    ax.set_title('Original - ' + image[7:11])

    ax = fig.add_subplot(4, 2, count+1)
    ax.imshow(img_mask, vmin=0)
    ax.set_title(image[7:11] + ' Forest Cover: ' +\
                  str(percent_forest(img_mask)) + '%')
    
    count += 2


# There we have it. The general cover of forest is decreasing with time.
# let us use the HSV color space to see if we will have better results.

# ## Color Segmentation in HSV

# We are now going to use the HSV color channel. We will be most
# interested in the green hue. HSV color space is nearer to how
# humans understand color.

# In[21]:


#Utility function to plot HSV histograms
def build_hist_hsv(img, title = '', yscale = 'linear'):
    assert(img.shape[2] == 3) #has 3 channels

    #Build the histograms
    hue_hist=cv2.calcHist([img], [0], None, [180], [0, 179])
    sat_hist=cv2.calcHist([img], [1], None, [256], [0, 255])
    val_hist=cv2.calcHist([img], [2], None, [256], [0, 255])
    
    fig = plt.figure(figsize=[20, 5])
    fig.suptitle(title)
    #Blue channel
    ax = fig.add_subplot(1, 3, 1)
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)
    ax.plot(hue_hist, color = 'b', label = 'Hue')
    ax.grid()
    ax.legend()
    
    #Green channel
    ax = fig.add_subplot(1, 3, 2)
    
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)
    
    ax.plot(sat_hist, color = 'g', label = 'Saturation')
    ax.grid()
    ax.legend()

    #Red channel
    ax = fig.add_subplot(1, 3, 3)
    
    #Setting y scale
    if(yscale == 'log'):
        ax.set_yscale(yscale, base=2)
    else:
        ax.set_yscale(yscale)

    ax.plot(val_hist, color = 'r', label = 'Value')
    ax.grid()
    ax.legend()


# In[22]:


#convert 1985 & 2011 to HSV and plot
image1985_hsv = cv2.cvtColor(image1985, cv2.COLOR_BGR2HSV)
image2011_hsv = cv2.cvtColor(image2011, cv2.COLOR_BGR2HSV)

#plotting their HSV values
build_hist_hsv(image1985_hsv, '1985 - HSV - Linear')
build_hist_hsv(image2011_hsv, '2011 - HSV - Linear')


# Looking at the hue, the value picks highest at around 50.
# We see that the peak is greater in 1985 than in 2011,
# at over 200K against 80K. We also see a smaller bump before
# the major peak in 1985. The bump becomes larger in the
# 2011 plot at over 40K.
# 
# Over at the value section, we note with interest that, for 2011,
# on top of the major peak at ~100, ther is a small bump rising
# from 150, with its highest at ~190.
# Let us plot the log plots to see.

# In[23]:


#plotting their HSV values
build_hist_hsv(image1985_hsv, '1985 - HSV - Linear', 'log')
build_hist_hsv(image2011_hsv, '2011 - HSV - Linear', 'log')


# We see that the second smaller bump in the Hue section is more
# significant than in the linear plot. We also see that the second
# bump in the Value channel is more pronounced in 2011 than in 1985.
# 
# We will cap our Hue at the considerable dip between 50 and 75,
# around 65. We can also cap our value at where the high peak
# starts falling, around 105.

# In[24]:


#Just like we did, we will also plot the images with their masks,
#We will also display the forest cover as a percentage of the whole mass
#We hope to comapre the ranges for RGB and HSV
images = glob.glob('images/*.jpg')
fig = plt.figure(figsize=[10, 15])
count = 1

for image in images:
    #read the image
    img_x = cv2.imread(image)

    #convert to HSV
    img_hsv = cv2.cvtColor(img_x, cv2.COLOR_BGR2HSV)

    #build a mask
    img_mask = build_masks(img_hsv, np.array([40, 0, 0]),\
                           np.array([65, 255, 105]))

    #display image and mask
    ax = fig.add_subplot(4, 2, count)
    ax.imshow(img_x[:, :, ::-1])
    ax.set_title('Original - ' + image[7:11])

    ax = fig.add_subplot(4, 2, count+1)
    ax.imshow(img_mask, vmin=0)
    ax.set_title(image[7:11] + ' Forest Cover: ' +\
                 str(percent_forest(img_mask)) + '%')
    
    count += 2


# The forest cover as we see here is smaller than in the
# BGR color space. However, the differences are minor and
# they all show a significant degree of change as the years progress.
# So, there we have it. Color segmentation with OpenCV.
