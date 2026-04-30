# Experiment 13
# Boundary and Regional Descriptors using Chain Code and Structural Features

import cv2
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------------
# Load Image
# -----------------------------------
image = cv2.imread("input13.jpg")

if image is None:
    print("Error: Image not found.")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# -----------------------------------
# Convert to Binary Image
# -----------------------------------
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

# -----------------------------------
# Find Contours (Object Boundary)
# -----------------------------------
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# Select largest contour
contour = max(contours, key=cv2.contourArea)

# Draw contour
boundary_img = image.copy()
cv2.drawContours(boundary_img, [contour], -1, (0,255,0), 2)

# -----------------------------------
# Chain Code Directions (8-connectivity)
# -----------------------------------
directions = {
    (1,0):0, (1,-1):1, (0,-1):2, (-1,-1):3,
    (-1,0):4, (-1,1):5, (0,1):6, (1,1):7
}

chain_code = []

pts = contour[:,0,:]

for i in range(len(pts)-1):
    dx = pts[i+1][0] - pts[i][0]
    dy = pts[i+1][1] - pts[i][1]

    dx = np.sign(dx)
    dy = np.sign(dy)

    if (dx,dy) in directions:
        chain_code.append(directions[(dx,dy)])

# -----------------------------------
# Regional Descriptors
# -----------------------------------
area = cv2.contourArea(contour)
perimeter = cv2.arcLength(contour, True)

x, y, w, h = cv2.boundingRect(contour)
aspect_ratio = float(w) / h

# -----------------------------------
# Display Results
# -----------------------------------
plt.figure(figsize=(10,6))

plt.subplot(1,2,1)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(cv2.cvtColor(boundary_img, cv2.COLOR_BGR2RGB))
plt.title("Boundary Descriptor")
plt.axis("off")

plt.tight_layout()
plt.show()

# -----------------------------------
# Output Features
# -----------------------------------
print("Chain Code (first 50 values):")
print(chain_code[:50])

print("\nRegional Descriptors:")
print("Area =", area)
print("Perimeter =", perimeter)
print("Aspect Ratio =", round(aspect_ratio, 2))

print("\nChain code represents object boundary direction changes.")
print("Regional descriptors describe size and shape properties.")