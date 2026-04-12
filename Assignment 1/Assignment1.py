
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Task 1: Welcome Message
print("====================================")
print("   Document Scanner Simulation")
print("   Image Processing Assignment")
print("====================================")
print("This program demonstrates image acquisition, sampling, quantization, and quality analysis.\n")

# Task 2: Image Acquisition
# Load image (Replace 'document.jpg' with your file)
image = cv2.imread('image3.jpg')

if image is None:
    print("Error: Image not found. Please check the file path.")
    exit()

# Resize to 512x512
image_resized = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)

# Task 3: Image Sampling (Resolution Analysis)
high_res = gray.copy()
medium_res = cv2.resize(gray, (256, 256))
low_res = cv2.resize(gray, (128, 128))

# Upscale back for comparison
medium_up = cv2.resize(medium_res, (512, 512))
low_up = cv2.resize(low_res, (512, 512))

# Task 4: Image Quantization
def quantize(image, levels):
    factor = 256 // levels
    quantized = (image // factor) * factor
    return quantized

quant_256 = gray  # original
quant_16 = quantize(gray, 16)
quant_4 = quantize(gray, 4)

# Task 5: Visualization
plt.figure(figsize=(12, 10))

# Original and grayscale
plt.subplot(4, 3, 1)
plt.title("Original Image")
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(4, 3, 2)
plt.title("Grayscale Image")
plt.imshow(gray, cmap='gray')
plt.axis('off')

# Sampling results
plt.subplot(4, 3, 4)
plt.title("High Resolution (512x512)")
plt.imshow(high_res, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 5)
plt.title("Medium Resolution (256→512)")
plt.imshow(medium_up, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 6)
plt.title("Low Resolution (128→512)")
plt.imshow(low_up, cmap='gray')
plt.axis('off')

# Quantization results
plt.subplot(4, 3, 7)
plt.title("256 Levels (8-bit)")
plt.imshow(quant_256, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 8)
plt.title("16 Levels (4-bit)")
plt.imshow(quant_16, cmap='gray')
plt.axis('off')

plt.subplot(4, 3, 9)
plt.title("4 Levels (2-bit)")
plt.imshow(quant_4, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Task 5: Observations
print("\n===== Observations =====")
print("1. As resolution decreases, fine text details become blurred.")
print("2. Low resolution images show poor edge sharpness.")
print("3. Quantization reduces gray levels, causing banding artifacts.")
print("4. Lower gray levels reduce readability of text.")
print("5. High resolution and higher gray levels are more suitable for OCR systems.")