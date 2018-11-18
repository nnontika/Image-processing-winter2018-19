# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

real_img = cv2.imread('bauckhage.jpg', 0)                 # read image
mask = np.copy(real_img)                              # copy original image to create mask
row, col = real_img.shape                             # retrieve number of rows and columns
c_row, c_col = int(row / 2), int(col / 2)             # determine center of rows and columns
r_max = 80                                            # user defined values for the band (range constants)
r_min = 30

# Fourier Transform
f_array = np.fft.fft2(real_img)                       # 2D fast Fourier transform to get frequency transform
f_shift = np.fft.fftshift(f_array)                    # move zero frequency to the center of the spectrum
fft = np.log10(abs(f_shift))                          # compute logarithm of the absolute value of the complex number

# Band pass filter
for i in range(row):
    for j in range(col):
        x = np.array((i, j))
        y = np.array((c_row, c_col))
        if r_min <= np.linalg.norm(x-y) <= r_max:     # if r_min ≤ ||(x; y) − (w/2 ; h/2 )|| ≤ r_max then 0
            mask[i][j] = 1                            # only the points living in concentric circle are ones for masking
        else:
            mask[i][j] = 0

# Inverse Fourier Transform
f_shift = f_shift * mask                              # apply mask with frequency transform
fi_shift = np.fft.ifftshift(f_shift)                  # move zero frequency to the top left corner
i_fft = np.fft.ifft2(fi_shift)                        # 2D Inverse fast Fourier transform
i_fft = np.abs(i_fft)                                 # get the absolute value of the complex number

# Visualization of all images
plt.subplot(2, 2, 1), plt.imshow(real_img, cmap='gray')
plt.title('Input Image'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(fft, cmap='gray')
plt.title('After FFT'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(mask, cmap='gray')
plt.title('After Band Pass Filter'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(i_fft, cmap='gray')
plt.title('After Inverse FFT'), plt.xticks([]), plt.yticks([])
plt.show()
