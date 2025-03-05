import cv2
import os
import numpy as np
import matplotlib.pyplot as plt

# Function to compute the average histogram across all images in the folder
def compute_average_histogram(image_files, input_folder):
    hist_sum = np.zeros(256)
    total_pixels = 0
    
    # Compute the Y-channel histogram for each image
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        y_channel = img_yuv[:,:,0]  # Extract Y channel
        
        # Calculate histogram
        hist = cv2.calcHist([y_channel], [0], None, [256], [0, 256])
        hist_sum += hist.flatten()
        total_pixels += y_channel.size
    
    # Compute the average histogram
    avg_hist = hist_sum / total_pixels
    return avg_hist

# Function to apply Dynamic Histogram Equalization (DHE) using the average histogram
def dynamic_hist_eq_avg(image, avg_hist):
    # Convert the input image to YUV color space
    img_yuv = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
    y_channel = img_yuv[:,:,0]  # Extract Y channel
    
    # Compute the cumulative distribution function (CDF) from the average histogram
    cdf = np.cumsum(avg_hist)  
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())  # Normalize CDF
    
    # Apply CDF to the Y channel using interpolation
    y_eq = np.interp(y_channel.flatten(), np.arange(256), cdf_normalized).reshape(y_channel.shape)
    
    # Update the Y channel and convert back to BGR color space
    img_yuv[:,:,0] = y_eq
    img_eq = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
    
    return img_eq

# Function to plot and save histograms for comparison
def plot_histograms(original_img, equalized_img, output_folder, image_file):
    img_yuv = cv2.cvtColor(original_img, cv2.COLOR_BGR2YUV)
    y_channel_original = img_yuv[:,:,0]
    hist_original = cv2.calcHist([y_channel_original], [0], None, [256], [0, 256])
    
    img_yuv_eq = cv2.cvtColor(equalized_img, cv2.COLOR_BGR2YUV)
    y_channel_equalized = img_yuv_eq[:,:,0]
    hist_equalized = cv2.calcHist([y_channel_equalized], [0], None, [256], [0, 256])
    
    # Plot histograms for original and equalized images
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(hist_original, color='gray')
    plt.title('Original Y Channel Histogram')
    plt.xlim(0, 256)
    
    plt.subplot(1, 2, 2)
    plt.plot(hist_equalized, color='blue')
    plt.title('Equalized Y Channel Histogram')
    plt.xlim(0, 256)
    
    plt.tight_layout()
    
    # Save the histogram plot
    hist_image_path = os.path.join(output_folder, os.path.basename(image_file).replace('.jpg', '_histogram.png'))
    plt.savefig(hist_image_path)
    plt.close()
    print(f"Histogram saved as {hist_image_path}")

# Function to apply DHE to all images in a folder and save results
def equalize_histograms_in_folder(input_folder, output_folder):
    # Get the list of image files in the input folder
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith('.jpg')]
    
    # Create output folder if it does not exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # 1. Compute the average histogram from all images
    avg_hist = compute_average_histogram(image_files, input_folder)
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        
        # Read the image
        img = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if img is None:
            continue
        
        # 2. Apply DHE using the computed average histogram
        img_eq = dynamic_hist_eq_avg(img, avg_hist)
        
        # Save the processed image to the output folder
        output_image_path = os.path.join(output_folder, image_file)
        cv2.imwrite(output_image_path, img_eq)
        print(f"Processed {image_file}")
        
        # Save histogram comparison plot
        plot_histograms(img, img_eq, output_folder, image_file)

# Define input and output folder paths
input_folder = r'C:\Users\MSI\Desktop\histogram_test\images'
output_folder = os.path.join(input_folder, 'dhe_modified')

# Apply DHE to all images in the folder
equalize_histograms_in_folder(input_folder, output_folder)
