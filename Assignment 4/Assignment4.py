
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Small font for clean output
plt.rcParams.update({'font.size': 6})

# Load Image
image = cv2.imread('traffic.jpg')

if image is None:
    print("Error: Image not found.")
    exit()

image = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Task 1: Edge Detection

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

# Canny
canny = cv2.Canny(gray, 100, 200)

# Task 2: Object Representation

# Use Canny edges for contour detection
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = image.copy()

areas = []
perimeters = []

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    # Filter small noise
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

        areas.append(area)
        perimeters.append(perimeter)

# Task 3: Feature Extraction (ORB)

orb = cv2.ORB_create()

keypoints, descriptors = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(image, keypoints, None, color=(0, 255, 0), flags=0)

# Task 4: Analysis

print("\n===== Edge Detection Comparison =====")
print("1. Sobel detects gradients but produces thicker edges.")
print("2. Canny provides thin, well-defined edges with noise suppression.")

print("\n===== Object Representation =====")
print(f"Total Objects Detected: {len(areas)}")
print(f"Average Area: {np.mean(areas) if areas else 0:.2f}")
print(f"Average Perimeter: {np.mean(perimeters) if perimeters else 0:.2f}")

print("\n===== Feature Extraction =====")
print(f"Number of Keypoints Detected: {len(keypoints)}")

print("\n===== Traffic Monitoring Insight =====")
print("1. Edge detection helps identify road boundaries and vehicles.")
print("2. Contours allow object counting (cars, bikes).")
print("3. ORB features help track objects across frames.")
print("4. Useful for traffic flow analysis and congestion detection.")

# Visualization

plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Sobel Edges")
plt.imshow(sobel, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Canny Edges")
plt.imshow(canny, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Contours & Bounding Boxes")
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("ORB Keypoints")
plt.imshow(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()