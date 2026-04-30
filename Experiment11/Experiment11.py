# Experiment 11
# Morphological Operations: Dilation and Erosion on Binary Image

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Image
# -----------------------------------
image = cv2.imread("input2.jpeg", 0)

if image is None:
    print("Error: Image not found.")
    exit()

# -----------------------------------
# Convert to Binary Image
# -----------------------------------
_, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Structuring Element (Kernel)
kernel = np.ones((5,5), np.uint8)

# -----------------------------------
# Morphological Operations
# -----------------------------------

# Erosion
erosion = cv2.erode(binary, kernel, iterations=1)

# Dilation
dilation = cv2.dilate(binary, kernel, iterations=1)

# Opening = Erosion + Dilation
opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

# Closing = Dilation + Erosion
closing = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

# -----------------------------------
# Display Results
# -----------------------------------
plt.figure(figsize=(12,8))

plt.subplot(2,3,1)
plt.imshow(binary, cmap='gray')
plt.title("Binary Image")
plt.axis("off")

plt.subplot(2,3,2)
plt.imshow(erosion, cmap='gray')
plt.title("Erosion")
plt.axis("off")

plt.subplot(2,3,3)
plt.imshow(dilation, cmap='gray')
plt.title("Dilation")
plt.axis("off")

plt.subplot(2,3,4)
plt.imshow(opening, cmap='gray')
plt.title("Opening")
plt.axis("off")

plt.subplot(2,3,5)
plt.imshow(closing, cmap='gray')
plt.title("Closing")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------------
# Explanation
# -----------------------------------
print("Erosion removes small white noise and shrinks objects.")
print("Dilation enlarges white regions and fills gaps.")
print("Opening removes noise while preserving shape.")
print("Closing fills holes and connects nearby objects.")