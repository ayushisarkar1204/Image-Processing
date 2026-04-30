# Experiment 3
# Image Enhancement using Contrast Stretching and Histogram Equalization

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image in grayscale
image = cv2.imread("image3.jpg", 0)

# Check image
if image is None:
    print("Error: Image not found.")
    exit()

# -----------------------------------
# Contrast Stretching
# -----------------------------------
min_val = np.min(image)
max_val = np.max(image)

contrast_stretch = ((image - min_val) / (max_val - min_val) * 255).astype(np.uint8)

# -----------------------------------
# Histogram Equalization
# -----------------------------------
hist_equalized = cv2.equalizeHist(image)

# -----------------------------------
# Display Images
# -----------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(contrast_stretch, cmap='gray')
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(hist_equalized, cmap='gray')
plt.title("Histogram Equalization")
plt.axis("off")

# -----------------------------------
# Histograms
# -----------------------------------
plt.subplot(2,3,4)
plt.hist(image.ravel(), 256, [0,256])
plt.title("Original Histogram")

plt.subplot(2,3,5)
plt.hist(contrast_stretch.ravel(), 256, [0,256])
plt.title("Contrast Stretch Histogram")

plt.subplot(2,3,6)
plt.hist(hist_equalized.ravel(), 256, [0,256])
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()

# -----------------------------------
# Explanation
# -----------------------------------
print("Contrast Stretching improves brightness range.")
print("Histogram Equalization redistributes intensity values.")
print("Both methods enhance image visibility and contrast.")