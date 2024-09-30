import cv2
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams['image.cmap'] = 'gray'

img = cv2.imread('images/bluebird.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(img)

bird_dft = np.fft.fft2(img)

bird_dft_plot = 20 * np.log(np.abs(bird_dft))

plt.imshow(bird_dft_plot)

bird_dft_shift = np.fft.fftshift(bird_dft)

bird_dft_shift_plot = 20 * np.log(np.abs(bird_dft_shift))

plt.imshow(bird_dft_shift_plot)

print(bird_dft_shift[0, 0], bird_dft_shift[0, 473])
print(bird_dft_shift[354, 0], bird_dft_shift[354, 473])

print(bird_dft[0, 0], bird_dft[0, 473])
print(bird_dft[354, 0], bird_dft[354, 473])

get_ipython().run_line_magic('pinfo2', 'np.fft.fft2')

Ts = 1/50
t = np.arange(0, 10, Ts)
x = np.sin(2 * np.pi * 15 * t) + np.sin(2 * np.pi * 20 * t) #sin(2 pi frequency time)
plt.plot(t, x)

y = np.fft.fft(x)
fs = 1/Ts
f = np.arange(0, len(y)) * fs/len(y) #k(index)/N(no of elements in fft) * R(sampling rate)

plt.plot(f, abs(y)) # abs(y) == magnitude == sqrt(re**2 + img**2)

xnoise = np.random.randn(len(y))
x = x + xnoise
plt.plot(t, x)

y_dirty = np.fft.fft(x)
y_dirty_shift = np.fft.fftshift(y_dirty)
fshift = np.arange(-len(x)/2, len(x)/2) * ((1/Ts) / len(x))
plt.plot(fshift, abs(y_dirty_shift))

y_clean = y_dirty.copy()
y_clean[abs(y_clean) < 100] = 0
x_clean = np.fft.ifft(y_clean)
plt.plot(t, x_clean)

#read an image
python = cv2.imread('images/python.bmp', cv2.IMREAD_GRAYSCALE)

python_fft = np.fft.fft2(python)

#let us plot the data
python_fft_shift = np.fft.fftshift(python_fft)
plt.imshow(np.log(abs(python_fft_shift) + 1), cmap='viridis')

# Create a new figure
plt.figure()

# Plot each complex number as an arrow
for num in python_fft_shift.flat:
    plt.arrow(0, 0, num.real, num.imag, head_width=0.1, head_length=0.2, fc='blue', ec='blue')

# Set the limits of the plot
plt.xlim(-4000, 4000)
plt.ylim(-4300, 4300)

# Add labels and a grid
plt.xlabel('Real')
plt.ylabel('Imaginary')
plt.grid(True)

# Show the plot
plt.show()

def plot_polar_image(image):
    """
    Given any image, we will produce the Argand diagram
    of the image.

    This function expects that 'image' is of complex dtype
    and is the result of a dft

    We will also find the ifft and plot as an image.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(12, 4)

    # Plot each complex number as an arrow
    for num in image.flat:
        ax1.arrow(0, 0, num.real, num.imag, head_width=0.1,
                  head_length=0.2, fc='blue', ec='blue')

    
    # Set the limits of the plot
    ax1.set_xlim(-5000, 5000);
    ax1.set_ylim(-5000, 5000);
    ax1.set_xlabel('Re');
    ax1.set_ylabel('Im');
    ax1.grid(True)

    image_ifft = np.fft.ifft2(image)
    
    ax2.imshow(abs(image_ifft))

plot_polar_image(python_fft)

def change_image(image, new_part, is_mag=0):
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

    plot_polar_image(new_image)        

magnitude = abs(python_fft_shift)
magnitude_change = magnitude * 1.5 * np.random.rand(magnitude.shape[0], magnitude.shape[1])

change_image(python_fft, magnitude_change)

magnitude_change