import os
from skimage.metrics import peak_signal_noise_ratio
from skimage.io import imread

def psnr_between_folders(folder1, folder2):
    psnr_values = []
    
    # Get list of filenames in folder1
    filenames = os.listdir(folder1)
    
    for filename in filenames:
        if filename.endswith(".jpg") or filename.endswith(".png"):
            # Read corresponding images from both folders
            img_path1 = os.path.join(folder1, filename)
            img_path2 = os.path.join(folder2, filename)
            img1 = imread(img_path1)
            img2 = imread(img_path2)
            
            # Compute PSNR between corresponding images
            psnr = peak_signal_noise_ratio(img1, img2)
            psnr_values.append(psnr)
    
    # Compute average PSNR across all images
    avg_psnr = sum(psnr_values) / len(psnr_values)

    print (len(psnr_values))
    
    return avg_psnr

# Example usage:
folder1 = "mp2_test/custom_test/sharp/"
folder2 = "mp2_test/custom_test/blur/"

avg_psnr = psnr_between_folders(folder1, folder2)
print(f"Average PSNR between corresponding images: {avg_psnr} dB")
