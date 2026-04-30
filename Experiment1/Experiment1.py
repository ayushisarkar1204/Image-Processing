# Experiment 1
# Implement a program to acquire and display an image
# Demonstrate image sensing and acquisition using OpenCV

import cv2
import matplotlib.pyplot as plt

# Step 1: Acquire Image
# Replace 'sample.jpg' with your image file path
image = cv2.imread('image.jpg')

# Check if image loaded successfully
if image is None:
    print("Error: Image not found. Please check file path.")
else:
    print("Image acquired successfully!")

    # Step 2: Convert BGR to RGB (for matplotlib display)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Step 3: Display Image
    plt.imshow(image_rgb)
    plt.title("Acquired Image")
    plt.axis("off")
    plt.show()

# Components of Image Processing System:
print("\nComponents of Image Processing System:")
print("1. Image Sensor (Camera/Scanner) - Captures image.")
print("2. Digitizer - Converts analog image to digital form.")
print("3. Processor - Performs image processing operations.")
print("4. Storage - Saves images.")
print("5. Display Device - Shows processed image.")
print("6. Software - Controls processing tasks.")