import numpy as np
import cv2
import matplotlib.pyplot as plt

plt.rcParams['image.cmap'] = 'gray'

#creating and weighting the kernel
ones_kernel = np.ones((3, 3), np.float32)
ones_kernel = ones_kernel/np.sum(ones_kernel)
ones_kernel

#creating the matrix
matrix = np.array([[ 36, 131, 251],
                      [203, 234, 228],
                      [ 79,  29, 153]])

#We now simulate a simple filter operation, a mean filter
value = np.uint8(np.sum(im_sample * ones_kernel))
print(value)
matrix[1, 1] = value

#load the image
dino = cv2.imread('images/dino.png', 0)
plt.imshow(dino)

#correlate
dino_corr = cv2.filter2D(dino, -1, ones_kernel)

flipp_kernel = cv2.flip(ones_kernel, -1)
print(flipp_kernel)

dino_conv = cv2.filter2D(dino, -1, flipp_kernel, anchor=(1, 1))

fig = plt.figure(figsize=[20, 5])
plt.subplot(131); plt.imshow(dino); plt.title('Original')
plt.subplot(132); plt.imshow(dino_corr); plt.title('Correlation')
plt.subplot(133); plt.imshow(dino_conv); plt.title('Convolution')

second_kernel = np.zeros((3, 3), np.uint8)
second_kernel[0, 0] = 1

equal = cv2.imread('images/equal.png', cv2.IMREAD_COLOR)

#correlation
eq_corr = cv2.filter2D(equal, -1, second_kernel)

second_flip = cv2.flip(second_kernel, -1)

eq_conv = cv2.filter2D(equal, -1, second_flip, anchor=(1, 1))

plt.figure(figsize=[15, 5])
plt.subplot(131); plt.imshow(equal[:, :, ::-1]); plt.title('Original')
plt.subplot(132); plt.imshow(eq_corr[:, :, ::-1]); plt.title('Correlation')
plt.subplot(133); plt.imshow(eq_conv[:, :, ::-1]); plt.title('Convolution')

icon = cv2.imread('images/python.bmp', cv2.IMREAD_COLOR)

#gaussian kernel
kern = np.array([[0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 1]], np.float32)

kern = kern / np.sum(kern)

icon_corr = cv2.filter2D(icon, -1, kern)

kern_flip = cv2.flip(kern, -1)
icon_conv = cv2.filter2D(icon, -1, kern_flip, anchor=(1, 1))

fig = plt.figure(figsize=[20, 5])
plt.subplot(131); plt.imshow(icon); plt.title('Original')
plt.subplot(132); plt.imshow(icon_corr); plt.title('Correlation')
plt.subplot(133); plt.imshow(icon_conv); plt.title('Convolution')

(icon_conv == icon_corr).all()