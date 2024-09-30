import cv2
import numpy as np
import matplotlib.pyplot as plt

checker = cv2.imread('images/checkerboard_color.png', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[3, 3])
plt.imshow(checker, cmap='gray')

xkernel = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])
sobel_xfiltered = cv2.filter2D(checker, cv2.CV_64F, xkernel)
plt.imshow(sobel_xfiltered, cmap='gray')

# the y-kernel is similar to the x-kernel rotated by 90
ykernel = np.rot90(xkernel)
sobel_yfiltered = cv2.filter2D(checker, cv2.CV_64F, ykernel)
plt.imshow(sobel_yfiltered, cmap='gray')

sobel_filtered = sobel_xfiltered + sobel_yfiltered
plt.imshow(sobel_filtered, cmap='gray')

_, sobel_thresh = cv2.threshold(sobel_filtered, -1, np.max(sobel_filtered), cv2.THRESH_BINARY_INV)
plt.imshow(sobel_thresh, cmap='gray')

sobelx = cv2.Sobel(checker, cv2.CV_64F, 1, 0, ksize=3) #dx, dy (1, 0)
sobely = cv2.Sobel(checker, cv2.CV_64F, 0, 1, ksize=3) #dx, dy (0, 1)
plt.figure(figsize=[15, 3])
plt.subplot(121); plt.imshow(sobelx, cmap='gray'); plt.title('X - kernel')
plt.subplot(122); plt.imshow(sobely, cmap='gray'); plt.title('Y - kernel')

sobel_full = np.sqrt(sobelx**2 + sobely**2)
plt.imshow(sobel_full, cmap='gray')

_, sobel_thresh = cv2.threshold(sobel_full, 1, np.max(sobel_full), cv2.THRESH_BINARY)
plt.imshow(sobel_thresh, cmap='gray')

xpadded = np.pad(xkernel, (398, 399))

xpadded_fft_shift = np.fft.fftshift(np.fft.fft2(xpadded))
checker_fft_shift = np.fft.fftshift(np.fft.fft2(checker))
plt.figure(figsize=[8, 4])
plt.subplot(121); plt.imshow(np.abs(xpadded_fft_shift), cmap='gray'); #plt.title('X - kernel')
plt.subplot(122); plt.imshow(np.abs(np.log(checker_fft_shift) + 1), cmap='gray'); #plt.title('Y - kernel #np,log is used for visibility

sobel_edge_fft = xpadded_fft_shift * checker_fft_shift
plt.figure(figsize=[4, 4])
plt.imshow(np.abs(np.log(sobel_edge_fft + (1/np.abs(np.max(sobel_edge_fft))**51))), cmap='gray') #to remove log(0) but also avoid much interference.

sobel_edge_ifft = np.fft.ifft2(np.fft.ifftshift(sobel_edge_fft))
plt.imshow(sobel_edge_ifft.real, cmap='gray')

np.unique(sobel_edge_ifft.real)

ypadded = np.pad(ykernel, (398, 399))
ypadded_fft_shift = np.fft.fftshift(np.fft.fft2(ypadded))
one_sobel_kernel = ypadded_fft_shift * xpadded_fft_shift
plt.imshow(np.abs(one_sobel_kernel), cmap='gray');

one_sobel_map = one_sobel_kernel * checker_fft_shift
final_sobel = np.fft.ifft2(np.fft.ifftshift(one_sobel_map))

#standardize
#final_sobel = final_sobel + np.abs(np.min(final_sobel))
plt.figure(figsize=[10, 5])
plt.subplot(121); plt.imshow(np.abs(np.log(one_sobel_map + (1/np.abs(np.max(one_sobel_map))**51))), cmap='gray');
plt.subplot(122); plt.imshow(np.abs(final_sobel), cmap='gray');

final_sobel.real[199, 499]

icon = cv2.imread('images/red_hibiscus.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(icon, cmap='gray')

icon_laplace = cv2.Laplacian(icon, cv2.CV_64F, ksize=3)

plt.imshow(icon_laplace, cmap='gray')

plt.imshow(icon_laplace[420:1000, 200:850], cmap='gray')

_, icon_laplace_thresh = cv2.threshold(icon_laplace, 35, np.max(icon_laplace), cv2.THRESH_BINARY)
plt.imshow(icon_laplace_thresh, cmap='gray')

# We are going to detect high-level edges from
# the flower above using the LoG operator.
#the gaussian low pass filter
import math
def distance(point1,point2):
    return math.sqrt((point1[0]-point2[0])**2 + (point1[1]-point2[1])**2)

def gaussianLP(D0,imgShape):
    base = np.zeros(imgShape[:2])
    rows, cols = imgShape[:2]
    center = (rows/2,cols/2)
    for x in range(cols):
        for y in range(rows):
            base[y,x] = math.exp(((-distance((y,x),center)**2)/(2*(D0**2))))
    return base

#For Gaussian Smoothing
def convolve(image, filter_used):
    img_in_freq_domain = np.fft.fft2(image)

    # Shift the zero-frequency component to the center of the frequency spectrum
    centered = np.fft.fftshift(img_in_freq_domain)

    # Multiply the filter with the centered spectrum
    filtered_image_in_freq_domain = centered * np.fft.fftshift(np.fft.fft2(filter_used))

    # Shift the zero-frequency component back to the top-left corner of the frequency spectrum
    inverse_fftshift_on_filtered_image = np.fft.ifft2(filtered_image_in_freq_domain)

    # Apply the inverse Fourier transform to obtain the final filtered image
    final_filtered_image = np.fft.ifftshift(inverse_fftshift_on_filtered_image)

    return np.abs(final_filtered_image)


# Above code was sourced from: https://medium.com/@mahmed31098/image-processing-with-python-frequency-\
# domain-filtering-for-noise-reduction-and-image-enhancement-d917e449db68 ).

#creating the Gaussian
sigma=10
gauss_flower = gaussianLP(sigma, icon.shape)
#We will now create a LoG for the Gaussian of the image
log_flower = cv2.Laplacian(gauss_flower, cv2.CV_64F, ksize=3)

plt.figure(figsize=[8, 8])
plt.subplot(121); plt.imshow(gauss_flower, cmap='gray'); plt.title(f'Gaussian, Sigma = {sigma}')
plt.subplot(122); plt.imshow(log_flower, cmap='gray'); plt.title('Laplacian of Gaussian')

# We will now convolve the LoG with the image
# The `filter2D` correlates but since the gaussian is
# reflected along the diagonal, this should be no
# problem.

# Getting the edge map
log_flower_edges = convolve(icon, log_flower)

#Thresholding to choose the zero-crossing for the edges
_, log_flower_zeros = cv2.threshold(log_flower_edges, 280, np.max(log_flower_edges), cv2.THRESH_BINARY)

#displaying
plt.figure(figsize=[7, 8])
plt.subplot(121); plt.imshow(log_flower_edges, cmap='gray'); plt.title('Edge Map')
plt.subplot(122); plt.imshow(log_flower_zeros, cmap='gray'); plt.title('Edges')

np.max(log_flower_edges)

# A function that creates a LoG operator, convolves with
# image and produces an output of the threshold image
def logedges(sigma, threshold_value):
    lpgaussian = gaussianLP(sigma, icon.shape[:2]) #low-pass gaussian
    logoperator = cv2.Laplacian(lpgaussian, cv2.CV_64F, ksize=3) #LoG operator
    
    #convolving
    edgemap = cv2.filter2D(icon, cv2.CV_64F, logoperator)
    #thresholding
    _, loc_edges = cv2.threshold(edgemap, threshold_value, np.max(edgemap), cv2.THRESH_BINARY)

    return edgemap


# In[30]:


edges_1 = logedges(1, 35)
edges_5 = logedges(5, 70)
edges_10 = logedges(10, 135)

plt.figure(figsize=[15, 10])
plt.subplot(131); plt.imshow(edges_1, cmap='gray'); plt.title('Sigma 1')
plt.subplot(132); plt.imshow(edges_5, cmap='gray'); plt.title('Sigma 5')
plt.subplot(133); plt.imshow(edges_10, cmap='gray'); plt.title('Sigma 10')


#loading the image
face = cv2.imread('images/face_2.jpg', cv2.IMREAD_GRAYSCALE)
plt.imshow(face, cmap='gray')

# getting the derivative of Gaussian
# the gradient magnitude
# the gradient orientation

def derivative_gaussian(img, sigma=1):
    #smoothing
    face_gaussian = convolve(img, gaussianLP(sigma, img.shape))
    
    #getting the x and y Sobel gradients
    Ix = cv2.Sobel(face_gaussian, cv2.CV_64F, 1, 0, ksize=3)
    Iy = cv2.Sobel(face_gaussian, cv2.CV_64F, 0, 1, ksize=3)

    magnitude = np.hypot(Ix, Iy)
    magnitude = magnitude/magnitude.max() * 255
    direction = np.arctan2(Ix, Iy)

    return magnitude, direction

edgemap_other, direction_map = derivative_gaussian(face, 2.2)
plt.imshow(edgemap_other, cmap='gray')
_ = plt.title('Gradient Magnitude')

# We will go through the image, at every point of grad.
# we will find the gradient

def edge_thinning_one(edgemap, orientmap):
    final = np.zeros(edgemap.shape, np.int32)
    for x in range(1, edgemap.shape[0] - 1):
        for y in range(1, edgemap.shape[1] - 1):
            p = 255
            q = 255
            #Get the angle
            theta = np.abs(np.rad2deg(orientmap[x, y]))
            if (0 <= theta < 22.5) or (157.5 <= theta <= 180):
                # 0, 180
                p = edgemap[x, y-1]
                q = edgemap[x, y+1]
            elif (22.5 <= theta < 67.5):
                # 45
                p = edgemap[x+1, y-1]
                q = edgemap[x-1, y+1]
            elif (67.5 <= theta < 112.5):
                # 90
                p = edgemap[x-1, y]
                q = edgemap[x+1, y]
            elif (112.5 <= theta < 157.5):
                # 135
                p = edgemap[x-1, y-1]
                q = edgemap[x+1, y+1]
            else:
                # For errors
                pass

            #check if the magnitude at point is greater than p,q
            if (edgemap[x, y] >= p) and (edgemap[x, y] >= q):
                final[x, y] = edgemap[x, y]
            else:
                final[x, y] = 0
    return final

thin_edges = edge_thinning_one(edgemap_other, direction_map)
plt.figure(figsize=[7, 7])
plt.imshow(thin_edges, cmap='gray')

def double_threshold(edgearray, lowbound = 40, highbound = 80):
    final_edge_array = np.zeros(edgearray.shape, np.uint8)
    
    #finding the edges
    #strong edges
    strong_edges_x, strong_edges_y = np.where(edgearray >= highbound)
    final_edge_array[strong_edges_x, strong_edges_y] = 255

    #weak edges
    weak_edges_x, weak_edges_y = np.where((edgearray >= lowbound) & (edgearray < highbound))
    final_edge_array[weak_edges_x, weak_edges_y] = highbound - ((highbound-lowbound)//2)
    
    return final_edge_array


threshold_image = double_threshold(thin_edges, 35, 50)
plt.figure(figsize=[10, 10])
plt.imshow(threshold_image, cmap='gray')

def hysteresis_thresh(edge_image):
    final_image = edge_image.copy()

    #local neighborhood
    row = np.array([-1, -1, -1, 0, 0, 0, 1, 1, 1])
    col = np.array([-1, 0, 1, -1, 0, 1, -1, 0, 1])
    
    for x in range(1, edge_image.shape[0] - 1):
        for y in range(1, edge_image.shape[1] - 1):
            #only the weak edges
            #if (edge_image[x, y] > 0) and (edge_image[x, y] < 255):
            max = 0
            for pos in range(len(row)):
                a = x + row[pos]
                b = y + col[pos]
                
                if edge_image[a, b] == 255:
                    max = edge_image[a, b]
        
            final_image[x, y] = max

    return final_image

final_piece = hysteresis_thresh(threshold_image)
plt.figure(figsize=[10, 10])
plt.imshow(final_piece, cmap='gray')

plt.imshow(cv2.Canny(face, 100, 200))

