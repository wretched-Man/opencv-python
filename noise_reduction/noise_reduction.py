#!/usr/bin/env python
# coding: utf-8

# # Removing Noise in Images using Median and Bilateral Filters

# We have seen linear filters such as the box filter and Gaussian Filter in image smoothing. We will now see how we can use filters to remove noise from an image. First, let us start by creating some noise in the image.

# We will create Gaussian, Uniform and Impulse ('salt and pepper') noise. We will then apply the various filters to see their effects on images. Some of the material in this section is sourced from [Kaggle](https://www.kaggle.com/code/chanduanilkumar/adding-and-removing-image-noise-in-python/notebook).

# In[225]:


import cv2
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'


# ### Gaussian Noise

# Also known as electronic noise. It is caused by the discrete nature of
# radiation of warm objects. To create this noise we use a normal distribution.

def gaussian_noise(img, mean, stddev, gamma = 1):
    #Create Gaussian Noise
    gauss_noise = np.zeros(img.shape[:2])
    cv2.randn(gauss_noise, mean, stddev)
    gauss_noise = (gauss_noise*gamma).astype(np.uint8)
    
    if len(img.shape) == 2:
        output = cv2.add(img, gauss_noise)
    elif len(img.shape) == 3:
        merged = cv2.merge([gauss_noise, gauss_noise, gauss_noise])
        output = cv2.add(img, merged)
    return gauss_noise, output
        

def gaussian_noise_b(img, mean, stddev, gamma = 1):
    gauss_noise = np.random.normal(mean/255, stddev/255, img.shape[:2]) * 255
    gauss_noise = gauss_noise + np.abs(gauss_noise.min())
    gauss_noise = (gauss_noise*gamma).astype(np.uint8)

    if len(img.shape) == 2:
        final = np.clip((gauss_noise + img), 0, 255).astype(np.uint8)
        return final
    elif len(img.shape) == 3:
        final = img.copy()
        for channel in range(img.shape[2]):
            final[:, :, channel] = np.clip((gauss_noise + img[:, :, channel]), 0, 255).astype(np.uint8)
            
        return gauss_noise, final
    else:
        return


# ### Uniform Noise

# As the name suggests, this type of noise follows a uniform distribution.
# It is caused by the quantization of image pixels to a number of discrete
# levels. To create a Uniform noise, we create a uniform distribution whose
# lower and upper bounds are the minimum and maximum pixel values
# (0 and 255 respectively) along the dimensions of the image. In this
# type of noise, the measured pixel image values are equally likely to be
# spread across a range of possible values either side of the true value.
# Hence, the PDF of the recorded value of each pixel value takes the form
# of a rectangle either side of the true value.

def uniform_noise(img, gamma=1):
    uni_noise = np.zeros(img.shape[:2])
    cv2.randu(uni_noise, 0, 256)
    uni_noise = (uni_noise * gamma).astype(np.uint8)
    
    if len(img.shape) == 2:
        output = cv2.add(img, uni_noise)
    elif len(img.shape) == 3:
        merged = cv2.merge([uni_noise, uni_noise, uni_noise])
        output = cv2.add(img, merged)
    return uni_noise, output


# ### Impulse / 'Salt and Pepper' Noise

# Impulse or "Salt and Pepper" noise is the sparse occurance of maximum (255)
# and minimum (0) pixel values in an image. This can be noticed as the presence
# of black pixels in bright regions and white pixels in dark regions. This type
# of noise is caused due to sharp and sudden disturbances in the image signal,
# and is mainly generated by errors in analog to digital conversion or bit transmission.
# 
# To create a Salt and Pepper noise, we first create a distribution similar
# to that used in Uniform noise and apply binary thresholding to create a
# grid of black and white pixels. The intensity of the noise can be easily
# altered by changing the threshold value.


def salt_pepper_noise(img, prob = 0.5):
    '''
    Add salt and pepper noise to image
    prob: Probability of the noise
    '''
    output = img.copy()
    sp_noise = np.zeros_like(img)
    if len(img.shape) == 2:
        black = 0
        white = 255            
    else:
        colorspace = img.shape[2]
        if colorspace == 3:  # RGB
            black = np.array([0, 0, 0], dtype='uint8')
            white = np.array([255, 255, 255], dtype='uint8')
        else:  # RGBA
            black = np.array([0, 0, 0, 255], dtype='uint8')
            white = np.array([255, 255, 255, 255], dtype='uint8')
    probs = np.random.random(output.shape[:2])
    output[probs < (prob / 2)] = black
    output[probs > 1 - (prob / 2)] = white
    sp_noise[probs < (prob / 2)] = black
    sp_noise[probs > 1 - (prob / 2)] = white
    return sp_noise, output


# (https://gist.github.com/gutierrezps/f4ddad3bbd2ad5a9b96e3c06378e28b4) link for the above.

# ### Noise in images

# Let us now look at how these types of noise can be recognized in images.

#load images
lena = cv2.imread("images/lena.jpg")
flower = cv2.imread("images/flower.jpg")

plt.figure(figsize=[15, 5])
plt.subplot(121); plt.imshow(lena[:, :, ::-1])
_ = plt.axis("off")

plt.subplot(122); plt.imshow(flower[:, :, ::-1])
_ = plt.axis("off")


def plot_noise_image(img, noise, noisy_img, noise_type=''):
    """
    utility function to plot the image, the noise and the resultant image.
    """
    plt.figure(figsize=[15, 5])
    plt.subplot(131); plt.imshow(img[:, :, ::-1]); plt.title('Original Image'); plt.axis("off")
    plt.subplot(132); plt.imshow(noise); plt.title(noise_type + ' Noise'); plt.axis("off")
    plt.subplot(133); plt.imshow(noisy_img[:, :, ::-1]); plt.title('Image with ' + noise_type + ' Noise'); plt.axis("off")


# gaussian noise
gauss_noise_lena, gauss_img_lena = gaussian_noise(lena, 128, 127, .3)
plot_noise_image(lena, gauss_noise_lena, gauss_img_lena, 'Gaussian')

# uniform noise
uni_noise_lena, uni_img_lena = uniform_noise(lena, .5)
plot_noise_image(lena, uni_noise_lena, uni_img_lena, 'Uniform')

imp_noise_lena, imp_img_lena = salt_pepper_noise(lena, .05)
plot_noise_image(lena, imp_noise_lena, imp_img_lena, 'Salt and Pepper')

# gaussian noise
gauss_noise_flower, gauss_img_flower = gaussian_noise(flower, 128, 127, .3)
plot_noise_image(flower, gauss_noise_flower, gauss_img_flower, 'Gaussian')

# uniform noise
uni_noise_flower, uni_img_flower = uniform_noise(flower, .6)
plot_noise_image(flower, uni_noise_flower, uni_img_flower, 'Uniform')

# Salt & pepper noise
imp_noise_flower, imp_img_flower = salt_pepper_noise(flower, .2)
plot_noise_image(flower, imp_noise_flower, imp_img_flower, 'Salt and Pepper')


# ### Filtering for Noise Removal

# ### Median Filter

# Median blur filtering is a nonlinear filtering technique that is most
# commonly used to remove salt-and-pepper noise from images. As the name
# suggests, salt-and-pepper noise shows up as randomly occurring white
# and black pixels that are sharply different from the surrounding.
# In color images, salt-and-pepper noise may appear as small random color spots.
# 
# The median filter sets the central pixel in a neighborhood as the median value
# of the pixels in the neighborhood. We can use OpenCV's `medianBlur()` for this.

# In[448]:


smooth_flower_median_imp = cv2.medianBlur(imp_img_flower, 5)
smooth_lena_median_imp = cv2.medianBlur(imp_img_lena, 3)


# In[451]:


fig, axs = plt.subplots(2, 3, figsize=[15, 9], layout='tight')

labels = ['Original', 'Image with Impulse Noise', 'Median Smoothed Image']
images = [lena, imp_img_lena, smooth_lena_median_imp, flower, imp_img_flower, smooth_flower_median_imp]

for pos, ax in enumerate(axs.flat):
    ax.imshow(images[pos][:, :, ::-1]); ax.set_title(labels[pos%3]); ax.axis('off')


# We can see that the median filter works very well with salt-and-pepper noise.

# ### Gaussian Blur

# Gaussian blur can also be used to remove noise.

# In[463]:


smooth_flower_gauss_gauss = cv2.GaussianBlur(gauss_img_flower, (11, 11), 20)
smooth_lena_gauss_gauss = cv2.GaussianBlur(gauss_img_lena, (5, 5), 11)

fig_2, axs_2 = plt.subplots(2, 3, figsize=[15, 9], layout='tight')

labels_gauss = ['Original', 'Image with Gaussian Noise', 'Gaussian Smoothed Image']
images_gauss = [lena, gauss_img_lena, smooth_lena_gauss_gauss, flower,\
                gauss_img_flower, smooth_flower_gauss_gauss]

for pos, ax in enumerate(axs_2.flat):
    ax.imshow(images_gauss[pos][:, :, ::-1]); ax.set_title(labels_gauss[pos%3]); ax.axis('off')


# Although we managed to remove some bit of noise from the images above,
# we can see that we have lost a considerable bit of detail in the process.
# We can better deal with this problem by using a blurring filter that
# smooths while preserving the edges.

# ### Bilateral Filter

# We are going to apply this bilateral edge-preserving filter to the images
# with gaussian noise above and see the difference.

# In[489]:


smooth_lena_gauss_bil = cv2.bilateralFilter(gauss_img_lena, 20, 120, 300)
smooth_flower_gauss_bil = cv2.bilateralFilter(gauss_img_flower, 20, 120, 300)

fig_3, axs_3 = plt.subplots(2, 3, figsize=[15, 9], layout='tight')

labels_bil = ['Image with Gaussian Noise', 'Filtered using Gaussian Filter', 'Filtered using Bilateral Filter']
images_bil = [gauss_img_lena, smooth_lena_gauss_gauss, smooth_lena_gauss_bil,\
                gauss_img_flower, smooth_flower_gauss_gauss, smooth_flower_gauss_bil]

for pos, ax in enumerate(axs_3.flat):
    ax.imshow(images_bil[pos][:, :, ::-1]); ax.set_title(labels_bil[pos%3]); ax.axis('off')


# We can see the difference that the bilateral filter brings to the table.
# It smooths the image while still retaining the edges. If the values of Sigma
# are enlarged further, the resultant image could have a comical effect to it.
# This is what is used in other camera image filters to smooth skin. However,
# it should be noted that the Bilateral filter can be quite slow compared to
# other smoothing filters.

# ### Using `fastNlMeansDenoising`

# We are going to use the OpenCV method, `fastNlMeansDenoising` to denoise the image.
# It expects an image with Gaussian white noise. Since our images are colored,
# we will use `fastNlMeansDenoisingColored`.

# In[505]:


smooth_lena_gauss_nl = cv2.fastNlMeansDenoisingColored(gauss_img_lena, None, 10, 10, 5, 19)

plt.figure(figsize=[15, 5])
plt.subplot(121); plt.imshow(gauss_img_lena[:, :, ::-1]); plt.title('Image with Gaussian Noise'); plt.axis("off")
plt.subplot(122); plt.imshow(smooth_lena_gauss_nl[:, :, ::-1]); plt.title('Denoised with non-local Denoising'); plt.axis("off")


# The results of the non-local denoiser are comparable to the bilateral filter.

# ### Filtering an image with Uniform Noise

# Since uniform noise basically means that a pixel is equally-likely to have as
# its value a range of values, we can use a mean filter to smooth out an image
# and reduce the noise. We set the central pixel as the mean of values in a neighborhood.
# This method, however, suffers from the same effect as the median filter, in that,
# edges are blurred too. We will also compare its result against the Bilateral
# filter over the same image.

# In[521]:


# 1st image smoothed with box filter and bilateral filter
smooth_lena_uni_mean = cv2.blur(uni_img_lena, (5, 5))
smooth_lena_uni_bil = cv2.bilateralFilter(uni_img_lena, 20, 130, 50)

# 2nd image smoothed with box filter and bilateral filter
smooth_flower_uni_mean = cv2.blur(uni_img_flower, (7, 7))
smooth_flower_uni_bil = cv2.bilateralFilter(uni_img_flower, 20, 130, 50)


fig_4, axs_4 = plt.subplots(2, 3, figsize=[15, 9], layout='tight')

labels_uni_bil = ['Image with Uniform Noise', 'Filtered using Mean Filter', 'Filtered using Bilateral Filter']
images_uni_bil = [uni_img_lena, smooth_lena_uni_mean, smooth_lena_uni_bil,\
                uni_img_flower, smooth_flower_uni_mean, smooth_flower_uni_bil]

for pos, ax in enumerate(axs_4.flat):
    ax.imshow(images_uni_bil[pos][:, :, ::-1]); ax.set_title(labels_uni_bil[pos%3]); ax.axis('off')


# We see that there is a considerable difference in image quality and noise
# reduction between the mean filter and bilateral filters. However, we could
# say that what the mean filter lacks in detail, it makes up for in speed.

# ### The Spectral differences between original and noisy images.

# We will now look at the Fourier transforms of the original and noisy images
# to see if there are any discernable differences in the Fourier space.
# We will have to work with our images in grayscale.

# In[534]:


def return_fft(img):
    '''Utility function that returns the fft of an img.
    Also, in this case, since the images are 3-channel, it
    will return a 1-channel image.'''

    if len(img.shape) > 2:
        # assume RGB
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    gray_fft = np.fft.fft2(img_gray)
    gray_fft_shift = np.fft.fftshift(gray_fft)

    #make suitable for plotting
    gray_fft_suit = np.abs(np.log(gray_fft_shift + 10**-35))

    return img_gray, gray_fft_suit


# In[542]:


images_lena = [lena, gauss_img_lena, uni_img_lena, imp_img_lena]
titles_lena = ['Original', 'With Gaussian Noise', 'With Uniform Noise', 'With Impulse Noise']

images_plot_gray = []
images_plot_fft = []

for image in images_lena:
    gray, fft = return_fft(image)
    images_plot_gray.append(gray)
    images_plot_fft.append(fft)


fig_5, axs_5 = plt.subplots(2, 4, figsize=[22, 12], layout='tight')

for pos, ax in enumerate(axs_5[0]):
    ax.imshow(images_plot_gray[pos]); ax.set_title(titles_lena[pos]); ax.axis('off')

for pos, ax in enumerate(axs_5[1]):
    ax.imshow(images_plot_fft[pos], cmap='viridis'); ax.set_title(titles_lena[pos] + ' FFT'); ax.axis('off')


# In[543]:


images_flower = [flower, gauss_img_flower, uni_img_flower, imp_img_flower]
titles_flower = ['Original', 'With Gaussian Noise', 'With Uniform Noise', 'With Impulse Noise']

images_plot_gray_f = []
images_plot_fft_f = []

for image in images_flower:
    gray, fft = return_fft(image)
    images_plot_gray_f.append(gray)
    images_plot_fft_f.append(fft)


fig_6, axs_6 = plt.subplots(2, 4, figsize=[22, 12], layout='tight')

for pos, ax in enumerate(axs_6[0]):
    ax.imshow(images_plot_gray_f[pos]); ax.set_title(titles_flower[pos]); ax.axis('off')

for pos, ax in enumerate(axs_6[1]):
    ax.imshow(images_plot_fft_f[pos], cmap='viridis'); ax.set_title(titles_flower[pos] + ' FFT'); ax.axis('off')


# We see that the high-frequency components of the original image are
# removed in the noisy images. Let us now look at how the frequency
# spectrum chnages as we increase the noise.

# ### Change in Frequency as Noise is Increased

# In[565]:


lena_5 = gaussian_noise(lena, 128, 127, .005)[1]
lena_25 = gaussian_noise(lena, 128, 127, .01)[1]
lena_60 = gaussian_noise(lena, 128, 127, .05)[1]
lena_100 = gaussian_noise(lena, 128, 127, .15)[1]

images_lena_gauss = [lena, lena_5, lena_25, lena_60, lena_100]
titles_lena_gauss = ['Original', '0.5% Gaussian Noise', '1% Gaussian Noise', '5% Gaussian Noise', '15% Gaussian Noise']

images_plot_gray_cf = []
images_plot_fft_cf = []

for image in images_lena_gauss:
    gray, fft = return_fft(image)
    images_plot_gray_cf.append(gray)
    images_plot_fft_cf.append(fft)


fig_7, axs_7 = plt.subplots(5, 2, figsize=[12, 25], layout='tight')

for pos, row in enumerate(axs_7):
    row[0].imshow(images_plot_gray_cf[pos]); row[0].set_title(titles_lena_gauss[pos]); row[0].axis('off')
    row[1].imshow(images_plot_fft_cf[pos], cmap='viridis'); row[1].set_title(titles_lena_gauss[pos] + ' FFT'); row[1].axis('off')


# In the above plot, we have created some Gaussian noise in the test
# image ranging from 0.5 - 15%. We quickly see from the frequency plot,
# that as the noise level increases, the high-frequency image components
# of the image are removed leaving us with only the low-freq. components. 

# ### Magnitude and Phase shift from Original

# We will now plot the magnitude and phase angle of the original image
# and compare it with the same plot of an image with 15% noise.

# In[575]:


#function to plot an image and its magnitude
def plot_polar_image(img, title=''):
    """
    Given any image, we will produce the Argand diagram
    of the image.
    """

    if len(img.shape) > 2:
        # assume RGB
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        img_gray = img.copy()

    gray_fft = np.fft.fft2(img_gray)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6 * 2, 4)
    fig.suptitle(title)
    
    ax1.imshow(img_gray), ax1.axis("off")

    # Plot each complex number as an arrow
    for num in gray_fft.flat:
        #plot as a ~1/10 of the original
        ax2.arrow(0, 0, num.real/3, num.imag/3, head_width=0.1,
                  head_length=0.2, fc='blue', ec='blue')

    
    # Set the limits of the plot
    ax2.set_xlim(-15000, 15000);
    ax2.set_ylim(-15000, 15000);
    ax2.set_xlabel('Re');
    ax2.set_ylabel('Im');
    ax2.grid(True)


# In[578]:


#100% gaussian noise
lena_100_true = gaussian_noise(lena, 128, 127)[1]
lena_100_small = cv2.resize(lena_100_true, None, fx=.25, fy=.25)


# In[579]:


lena_small = cv2.resize(lena, None, fx=.25, fy=.25)
lena_15_small = cv2.resize(lena_100, None, fx=.25, fy=.25)
plot_polar_image(lena_small, 'Original')
plot_polar_image(lena_15_small, '15% Gaussian Noise')
plot_polar_image(lena_100_small, '100% Gaussian Noise')


# What we can see from the plot above is that there is a significant
# change in the image magnitude, while the phase angle remains relatively the same.