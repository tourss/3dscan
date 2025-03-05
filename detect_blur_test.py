import cv2
import numpy as np
import os
import shutil

# Set paths
input_folder = r"C:\Users\MSI\Desktop\blur_test\images"
output_folder = os.path.join(input_folder, "no_blur")

# Create folder if it does not exist
os.makedirs(output_folder, exist_ok=True)

# Blur detection function (Laplacian method)
def is_not_blurry(image_path, threshold=60):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    laplacian_var = cv2.Laplacian(image, cv2.CV_64F).var()
    return laplacian_var > threshold  # If the value is large, the image is sharp

# Check all PNG and JPG files in the folder
for filename in os.listdir(input_folder):
    if filename.lower().endswith((".png", ".jpg")):
        file_path = os.path.join(input_folder, filename)

        # Check for blur and copy if not blurry
        if is_not_blurry(file_path):
            shutil.copy(file_path, os.path.join(output_folder, filename))
            print(f"{filename} copied to no_blur folder")

print("no_blur file copy is complete!")
