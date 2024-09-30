import cv2
import glob
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

#Read and display the images
images = glob.glob('.\\images/pexels-*.jpg')
read_images = []

for image in images:
    image = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    read_images.append(cv2.resize(image, None, fx=0.06, fy=0.06, interpolation=cv2.INTER_AREA))

# display the images
plt.figure(figsize=[5*3, 5])
plt.subplot(131); plt.imshow(read_images[0], cmap='gray'); plt.xticks=[]; plt.yticks=[]; plt.axis('off')
plt.subplot(132); plt.imshow(read_images[1], cmap='gray'); plt.xticks=[]; plt.yticks=[]; plt.axis('off')
plt.subplot(133); plt.imshow(read_images[2], cmap='gray'); plt.xticks=[]; plt.yticks=[]; plt.axis('off')

# We now compute their average magnitude and phase
avg_magnitude = np.zeros(read_images[0].shape)
avg_phase = np.zeros(read_images[0].shape)
for copy_img in read_images:
    copy_img = copy_img - np.mean(copy_img)
    img_fft = np.fft.fft2(copy_img)
    avg_magnitude = avg_magnitude + (abs(img_fft)/3)
    avg_phase = avg_phase + (np.angle(img_fft)/3)

#function to plot an image and its magnitude
def plot_polar_image(image, title=''):
    """
    Given any image, we will produce the Argand diagram
    of the image.

    This function expects that 'image' is of complex dtype
    and is the result of a dft

    We will also find the ifft and plot as an image.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(6 * 2, 4)
    fig.suptitle(title)

    image_ifft = np.fft.ifft2(image)
    ax1.imshow(abs(image_ifft))

    # Plot each complex number as an arrow
    for num in image.flat:
        #plot as a 1/4 of the original
        ax2.arrow(0, 0, num.real/2, num.imag/2, head_width=0.1,
                  head_length=0.2, fc='blue', ec='blue')

    
    # Set the limits of the plot
    ax2.set_xlim(-5000, 5000);
    ax2.set_ylim(-5000, 5000);
    ax2.set_xlabel('Re');
    ax2.set_ylabel('Im');
    ax2.grid(True)

# function to change either the magnitude or phase of
# an image and plot the resulting image
def change_image(image, new_part, is_mag=0, title=''):
    """
    This function takes an image, changes the image's
    magnitude/phase and displays the image.

    The shapes of image and new_part must be equal
    Image must be a complex - result of an fft

    image - a complex array, a result of fft
    new_part - either new magnitude or phase with which
        to change the image into
    is_mag - if new_part is magnitude, set to 0,
    else if new_part is phase, set to non-zero

    Calls plot_polar_image on result
    """

    #Split the current image into magnitude and phase
    magnitude = abs(image)
    phase = np.angle(image)

    new_image = np.empty(image.shape, np.complex128)
    if is_mag == 0:
        #This means we are to swap magnitude
        #Make a complex number Data[...,0] + 1j * Data[...,1]
        new_image = (new_part * np.cos(phase)) + 1j * (new_part * np.sin(phase))
    else:
        #Swapping phase
        new_image = (magnitude * np.cos(new_part)) + 1j * (magnitude * np.sin(new_part))

    plot_polar_image(new_image, title)

#the fft of all images
images_fft = []
for image in read_images:
    img_fft_shift = np.fft.fft2(image)
    images_fft.append(img_fft_shift)

#put it all in a convenience function to avoid repetition
def plot_changes(pos):
    plot_polar_image(images_fft[pos], 'Image ' + str(pos + 1) + ' - Original')
    change_image(images_fft[pos], avg_magnitude, title= 'Image ' + str(pos + 1) + ' - Average Magnitude')
    change_image(images_fft[pos], avg_phase, is_mag= 1, title= 'Image ' + str(pos + 1) + ' - Average Phase')

plot_changes(0)

plot_changes(1)

plot_changes(2)
