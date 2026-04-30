# Experiment 2
# Sampling and Quantization on an Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load image
image = cv2.imread("image2.jpg", 0)   # Load in grayscale

# Check image
if image is None:
    print("Error: Image not found.")
    exit()

# -----------------------------
# Function for Sampling
# -----------------------------
def sampling(img, scale):
    h, w = img.shape
    new_h = int(h * scale)
    new_w = int(w * scale)
    sampled = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
    return sampled

# -----------------------------
# Function for Quantization
# -----------------------------
def quantization(img, levels):
    step = 256 // levels
    quantized = (img // step) * step
    return quantized

# Different sampling rates
sample1 = sampling(image, 1.0)   # Original
sample2 = sampling(image, 0.5)   # 50%
sample3 = sampling(image, 0.25)  # 25%

# Different quantization levels
quant1 = quantization(image, 256)   # Original
quant2 = quantization(image, 16)    # 16 levels
quant3 = quantization(image, 4)     # 4 levels

# -----------------------------
# Display Results
# -----------------------------
plt.figure(figsize=(12,8))

# Sampling Results
plt.subplot(2,3,1)
plt.imshow(sample1, cmap='gray')
plt.title("Original Sampling")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(sample2, cmap='gray')
plt.title("50% Sampling")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(sample3, cmap='gray')
plt.title("25% Sampling")
plt.axis("off")

# Quantization Results
plt.subplot(2,3,4)
plt.imshow(quant1, cmap='gray')
plt.title("256 Levels")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(quant2, cmap='gray')
plt.title("16 Levels")
plt.axis("off")

plt.subplot(2,3,6)
plt.imshow(quant3, cmap='gray')
plt.title("4 Levels")
plt.axis("off")

plt.tight_layout()
plt.show()

# Explanation
print("Sampling reduces image resolution.")
print("Lower sampling rate = less detail, more pixelation.")
print("Quantization reduces gray levels.")
print("Lower quantization levels = more intensity loss, banding effect.")