import numpy as np
import matplotlib.pyplot as plt
import cv2

#Let us create a wave by adding four waves together
fig = plt.figure(figsize=[20, 5])
Ts = 1/50 #time between samples
t = np.arange(0, 10, Ts) #10 seconds... 1/Ts samples per second
x = np.sin(2*np.pi*8*t + 2*np.pi/5) + np.sin(2*np.pi*24*t + np.pi/2) + np.sin(2*np.pi*19*t + np.pi/3) + np.sin(2*np.pi*16*t + 3*np.pi/2)
plt.plot(t, x)

y = np.fft.fft(x) #1D fft
y_shift = np.fft.fftshift(y)
fs = 1/Ts
f = np.arange(-len(y)//2, len(y)//2) * fs/len(y) #k(index)/N(no of elements in fft) * R(sampling rate)

#A plot of phase against frequency
phase_ticks = ['-$$pi$$/2', '-$$pi$$/4', '0', '$$pi$$/4', '$$pi$$/4']
phase = np.linspace(-np.pi, np.pi, len(f))

plt.plot(f, (abs(y_shift)))
plt.title('Magnitude against Frequency')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')

plt.plot(f, np.angle(y_shift))
plt.title('Phase against Frequency')
plt.xlabel('Frequency')
plt.ylabel('Phase')


#read an image
python = cv2.imread('images/python.bmp', cv2.IMREAD_GRAYSCALE)

plt.figure(figsize=[3, 3])
plt.imshow(python, cmap='gray')

python_fft = np.fft.fft2(python)

#we shift the result to bring the zero frequencies to the centre
python_fft_shift = np.fft.fftshift(python_fft)

#we then plot
plt.figure(figsize=[3, 3])
plt.imshow(np.log(abs(python_fft_shift) + 1), cmap='gray')

#let us have a snippet of the X wave
print(y[0])
#a snippet of the image also
print(python_fft[0, 0])