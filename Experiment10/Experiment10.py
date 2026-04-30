# Experiment 10
# Image Segmentation using Edge Detection, Edge Linking and Boundary Detection

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Image
# -----------------------------------
image = cv2.imread("images1.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------------
# Edge Detection using Canny
# -----------------------------------
edges = cv2.Canny(gray, 100, 200)

# -----------------------------------
# Edge Linking using Morphological Closing
# -----------------------------------
kernel = np.ones((3,3), np.uint8)
linked_edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

# -----------------------------------
# Boundary Detection using Contours
# -----------------------------------
contours, _ = cv2.findContours(linked_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw boundaries on original image
boundary_image = image.copy()
cv2.drawContours(boundary_image, contours, -1, (0,255,0), 2)

# -----------------------------------
# Display Results
# -----------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(2,2,2)
plt.imshow(edges, cmap='gray')
plt.title("Canny Edge Detection")
plt.axis("off")

plt.subplot(2,2,3)
plt.imshow(linked_edges, cmap='gray')
plt.title("Edge Linking")
plt.axis("off")

plt.subplot(2,2,4)
plt.imshow(cv2.cvtColor(boundary_image, cv2.COLOR_BGR2RGB))
plt.title("Boundary Detection / Segmentation")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------------
# Explanation
# -----------------------------------
print("Canny operator detects edges in the image.")
print("Morphological closing links broken edges.")
print("Contours detect object boundaries for segmentation.")
print("Objects are segmented based on closed boundaries.")