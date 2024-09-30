# Artistic Features

import cv2
import numpy as np
import matplotlib.pyplot as plt


#load sample images
flower    = cv2.imread('images/flowers.jpg')
house     = cv2.imread('images/house.jpg')
new_york  = cv2.imread('images/newyork.jpg')

def display(img, filter_img, filter=''):
    """
        A convenience method to display out image.
    """
    
    plt.figure(figsize=[15, 7])
    
    plt.subplot(121)
    plt.axis('off')
    
    if len(img.shape) == 2:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img[:, :, ::-1])
    
    plt.title('Original Image');

    plt.subplot(122)
    plt.axis('off')
    
    if len(img.shape) == 2:
        plt.imshow(filter_img, cmap='gray');
    else:
        plt.imshow(filter_img[:, :, ::-1]);

    if filter == '':
        plt.title('Filtered Image');
    else:
        plt.title(filter);


# ## Vignette Effect

# It is made by use of a Gaussian Kernel of the same shape as the image

#creating the separable Gaussian
def vignette(img, level = 2):
    width, height = img.shape[:2]
    x_kernel = cv2.getGaussianKernel(width, width/level)
    y_kernel = cv2.getGaussianKernel(height, height/level)

    #getting the outer product
    gauss_kernel = x_kernel * y_kernel.T
    #normalize
    gauss_kernel = gauss_kernel /gauss_kernel.max()

    img_vignette = img.copy()
    
    for i in range(3):
        img_vignette[:, :, i] = img_vignette[:, :, i] * gauss_kernel
        
    return img_vignette

vignette_res = vignette(flower, 2)
display(flower, vignette_res, 'Vignette filter')

def sepia(img):
    img_sepia = img.copy()
    # Converting to RGB as sepia matrix below is for RGB.
    img_sepia = cv2.cvtColor(img_sepia, cv2.COLOR_BGR2RGB) 
    img_sepia = np.array(img_sepia, dtype = np.float64)
    img_sepia = cv2.transform(img_sepia, np.matrix([[0.393, 0.769, 0.189],
                                                    [0.349, 0.686, 0.168],
                                                    [0.272, 0.534, 0.131]]))
    # Clip values to the range [0, 255].
    sepia_copy = np.clip(img_sepia, 0, 255)
    sepia_copy = np.array(sepia_copy, dtype = np.uint8)
    sepia_copy = cv2.cvtColor(sepia_copy, cv2.COLOR_RGB2BGR)
    return sepia_copy

ny_sepia = sepia(new_york)
display(new_york, ny_sepia)