# Experiment 12
# Threshold-Based and Region-Based Image Segmentation

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Image
# -----------------------------------
image = cv2.imread("images5.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------------
# 1. Threshold-Based Segmentation
# -----------------------------------
_, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# -----------------------------------
# 2. Region-Based Segmentation
# Using Connected Components
# -----------------------------------
_, region = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
num_labels, labels = cv2.connectedComponents(region)

# Create colored output for regions
region_output = np.zeros((labels.shape[0], labels.shape[1], 3), dtype=np.uint8)

for i in range(1, num_labels):
    color = np.random.randint(0, 255, size=3)
    region_output[labels == i] = color

# -----------------------------------
# Display Results
# -----------------------------------
plt.figure(figsize=(12,8))

plt.subplot(1,3,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,3,2)
plt.imshow(thresh, cmap='gray')
plt.title("Threshold Segmentation")
plt.axis("off")

plt.subplot(1,3,3)
plt.imshow(region_output)
plt.title("Region-Based Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------------
# Comparison
# -----------------------------------
print("Thresholding is simple and fast.")
print("Best for images with clear foreground/background contrast.")
print("Region-based segmentation groups connected pixels.")
print("Better for separating multiple objects and complex regions.")