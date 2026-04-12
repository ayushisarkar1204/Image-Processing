
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Small font for clean output
plt.rcParams.update({'font.size': 6})

# Task 1: Image Compression (RLE)

# Load grayscale image
image = cv2.imread('medical.jpeg', cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Error: Image not found.")
    exit()

image = cv2.resize(image, (512, 512))

# Run Length Encoding
def rle_encode(img):
    pixels = img.flatten()
    encoded = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1

    encoded.append((prev, count))
    return encoded

encoded = rle_encode(image)

# Compression calculations
original_size = image.size          # number of pixels
compressed_size = len(encoded) * 2  # (value, count)

compression_ratio = original_size / compressed_size
storage_saving = (1 - (compressed_size / original_size)) * 100

print("\n===== Compression Results =====")
print(f"Original Size: {original_size}")
print(f"Compressed Size: {compressed_size}")
print(f"Compression Ratio: {compression_ratio:.2f}")
print(f"Storage Saving: {storage_saving:.2f}%")

# Task 2: Image Segmentation

# Global Thresholding
_, global_thresh = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# Otsu Thresholding
_, otsu_thresh = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Task 3: Morphological Processing

kernel = np.ones((3, 3), np.uint8)

# Apply on Otsu result (better segmentation generally)
dilation = cv2.dilate(otsu_thresh, kernel, iterations=1)
erosion = cv2.erode(otsu_thresh, kernel, iterations=1)

# Task 4: Analysis

print("\n===== Analysis =====")
print("1. RLE compression works well when large regions have similar intensity.")
print("2. Medical images with uniform background achieve better compression.")
print("3. Otsu’s thresholding provides automatic and more accurate segmentation.")
print("4. Global thresholding may fail under varying illumination.")
print("5. Dilation expands detected regions, useful for highlighting structures.")
print("6. Erosion removes small noise and refines boundaries.")
print("7. Segmented regions can represent tumors, bones, or organs depending on modality.")

# Visualization

plt.figure(figsize=(10, 8))

plt.subplot(2, 3, 1)
plt.title("Original Image")
plt.imshow(image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Global Threshold")
plt.imshow(global_thresh, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Otsu Threshold")
plt.imshow(otsu_thresh, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Dilation")
plt.imshow(dilation, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Erosion")
plt.imshow(erosion, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()