
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Small font for clean visualization
plt.rcParams.update({'font.size': 6})


# Task 1: System Overview

print("========================================")
print(" Intelligent Image Processing System ")
print("========================================")
print("This system performs preprocessing, enhancement, segmentation,")
print("feature extraction, and evaluation on an input image.\n")

# Task 2: Image Acquisition

image = cv2.imread('image6.jpg')

if image is None:
    print("Error: Image not found.")
    exit()

image = cv2.resize(image, (512, 512))
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


# Task 3: Enhancement & Restoration


# Noise functions
def add_gaussian(img, mean=0, sigma=25):
    noise = np.random.normal(mean, sigma, img.shape)
    noisy = img + noise
    return np.uint8(np.clip(noisy, 0, 255))

def add_sp(img, prob=0.02):
    noisy = img.copy()
    total = img.size

    num_salt = int(prob * total / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    num_pepper = int(prob * total / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

gaussian_noisy = add_gaussian(gray)
sp_noisy = add_sp(gray)

# Restoration filters
mean_filter = cv2.blur(gaussian_noisy, (3, 3))
median_filter = cv2.medianBlur(sp_noisy, 3)
gaussian_filter = cv2.GaussianBlur(gaussian_noisy, (3, 3), 0)

# Enhancement (CLAHE)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
enhanced = clahe.apply(gray)


# Task 4: Segmentation & Morphology

_, global_thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
_, otsu_thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

kernel = np.ones((3, 3), np.uint8)
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)


# Task 5: Object Representation


# Edge Detection
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(np.clip(sobel, 0, 255))

canny = cv2.Canny(gray, 100, 200)

# Contours
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_img = image.copy()

for cnt in contours:
    if cv2.contourArea(cnt) > 100:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x, y), (x + w, y + h), (0, 255, 0), 1)

# ORB Features
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
feature_img = cv2.drawKeypoints(image, kp, None, color=(0, 255, 0))


# Task 6: Performance Evaluation


def mse(img1, img2):
    return np.mean((img1 - img2) ** 2)

def psnr(img1, img2):
    m = mse(img1, img2)
    if m == 0:
        return float('inf')
    return 10 * math.log10((255 ** 2) / m)

def ssim(img1, img2):
    C1 = 6.5025
    C2 = 58.5225

    mu1 = img1.mean()
    mu2 = img2.mean()

    sigma1 = img1.var()
    sigma2 = img2.var()
    sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()

    return ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1**2 + mu2**2 + C1) * (sigma1 + sigma2 + C2))

print("\n===== Performance Metrics =====")
print(f"MSE (Original vs Enhanced): {mse(gray, enhanced):.2f}")
print(f"PSNR (Original vs Enhanced): {psnr(gray, enhanced):.2f} dB")
print(f"SSIM (Original vs Enhanced): {ssim(gray, enhanced):.4f}")

print(f"MSE (Original vs Restored): {mse(gray, gaussian_filter):.2f}")
print(f"PSNR (Original vs Restored): {psnr(gray, gaussian_filter):.2f} dB")
print(f"SSIM (Original vs Restored): {ssim(gray, gaussian_filter):.4f}")

# Task 7: Final Visualization

plt.figure(figsize=(12, 10))

plt.subplot(3, 3, 1)
plt.title("Original")
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 3, 2)
plt.title("Gaussian Noise")
plt.imshow(gaussian_noisy, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 3)
plt.title("Restored")
plt.imshow(gaussian_filter, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 4)
plt.title("Enhanced (CLAHE)")
plt.imshow(enhanced, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 5)
plt.title("Otsu Segmentation")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 6)
plt.title("Morphology")
plt.imshow(dilation, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 7)
plt.title("Sobel")
plt.imshow(sobel, cmap='gray')
plt.axis('off')

plt.subplot(3, 3, 8)
plt.title("Canny + Contours")
plt.imshow(cv2.cvtColor(contour_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(3, 3, 9)
plt.title("ORB Features")
plt.imshow(cv2.cvtColor(feature_img, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.tight_layout()
plt.show()

# Final Conclusion
print("\n===== Conclusion =====")
print("1. CLAHE significantly improves contrast in low-light images.")
print("2. Median filter is effective for salt-and-pepper noise.")
print("3. Otsu thresholding provides better segmentation than global threshold.")
print("4. Morphological operations refine object boundaries.")
print("5. Canny edge detector gives precise edges.")
print("6. ORB efficiently extracts feature points for object tracking.")
print("7. Overall, the system effectively processes and analyzes images.")