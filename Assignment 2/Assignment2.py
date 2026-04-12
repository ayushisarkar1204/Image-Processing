
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math

# Set very small font size globally
plt.rcParams.update({'font.size': 6})

# Task 1: Image Selection and Preprocessing
image = cv2.imread('image6.jpg')

if image is None:
    print("Error: Image not found.")
    exit()

# Resize for consistency
image = cv2.resize(image, (512, 512))

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Task 2: Noise Modeling

# Gaussian Noise
def add_gaussian_noise(img, mean=0, sigma=25):
    gaussian = np.random.normal(mean, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + gaussian
    noisy = np.clip(noisy, 0, 255)
    return noisy.astype(np.uint8)

# Salt & Pepper Noise
def add_salt_pepper(img, prob=0.02):
    noisy = img.copy()
    total_pixels = img.size

    # Salt noise
    num_salt = int(prob * total_pixels / 2)
    coords = [np.random.randint(0, i - 1, num_salt) for i in img.shape]
    noisy[coords[0], coords[1]] = 255

    # Pepper noise
    num_pepper = int(prob * total_pixels / 2)
    coords = [np.random.randint(0, i - 1, num_pepper) for i in img.shape]
    noisy[coords[0], coords[1]] = 0

    return noisy

gaussian_noisy = add_gaussian_noise(gray)
sp_noisy = add_salt_pepper(gray)

# Task 3: Image Restoration

# Filters for Gaussian noise
mean_gaussian = cv2.blur(gaussian_noisy, (3, 3))
median_gaussian = cv2.medianBlur(gaussian_noisy, 3)
gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (3, 3), 0)

# Filters for Salt & Pepper noise
mean_sp = cv2.blur(sp_noisy, (3, 3))
median_sp = cv2.medianBlur(sp_noisy, 3)
gaussian_sp = cv2.GaussianBlur(sp_noisy, (3, 3), 0)

# Task 4: Performance Evaluation

def mse(original, restored):
    return np.mean((original - restored) ** 2)

def psnr(original, restored):
    m = mse(original, restored)
    if m == 0:
        return float('inf')
    return 10 * math.log10((255 ** 2) / m)

# Compute metrics
results = {
    "Gaussian Noise": {
        "Mean Filter": (mse(gray, mean_gaussian), psnr(gray, mean_gaussian)),
        "Median Filter": (mse(gray, median_gaussian), psnr(gray, median_gaussian)),
        "Gaussian Filter": (mse(gray, gaussian_filtered), psnr(gray, gaussian_filtered)),
    },
    "Salt & Pepper Noise": {
        "Mean Filter": (mse(gray, mean_sp), psnr(gray, mean_sp)),
        "Median Filter": (mse(gray, median_sp), psnr(gray, median_sp)),
        "Gaussian Filter": (mse(gray, gaussian_sp), psnr(gray, gaussian_sp)),
    }
}

# Task 5: Analytical Discussion
print("\n===== Performance Comparison =====")
for noise_type, filters in results.items():
    print(f"\n--- {noise_type} ---")
    for filter_name, (m, p) in filters.items():
        print(f"{filter_name}: MSE = {m:.2f}, PSNR = {p:.2f} dB")

print("\n===== Analysis =====")
print("1. Gaussian noise is best reduced by Gaussian filter due to smoothing effect.")
print("2. Median filter performs best for salt-and-pepper noise as it removes impulse noise.")
print("3. Mean filter reduces noise but blurs edges significantly.")
print("4. Median filter preserves edges better than mean and Gaussian filters.")
print("5. Higher PSNR indicates better restoration quality.")

# Visualization
plt.figure(figsize=(12, 10))

# Original & Noisy Images
plt.subplot(3, 4, 1)
plt.title("Original")
plt.imshow(gray, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 2)
plt.title("Gaussian Noise")
plt.imshow(gaussian_noisy, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.title("Salt & Pepper Noise")
plt.imshow(sp_noisy, cmap='gray')
plt.axis('off')

# Gaussian Noise Restoration
plt.subplot(3, 4, 5)
plt.title("Mean (Gaussian)")
plt.imshow(mean_gaussian, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 6)
plt.title("Median (Gaussian)")
plt.imshow(median_gaussian, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.title("Gaussian Filter")
plt.imshow(gaussian_filtered, cmap='gray')
plt.axis('off')

# Salt & Pepper Restoration
plt.subplot(3, 4, 9)
plt.title("Mean (S&P)")
plt.imshow(mean_sp, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 10)
plt.title("Median (S&P)")
plt.imshow(median_sp, cmap='gray')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.title("Gaussian (S&P)")
plt.imshow(gaussian_sp, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()