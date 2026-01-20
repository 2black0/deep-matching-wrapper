import cv2
import numpy as np

# Ensure you have an image named 'input_image.jpg' in the same directory
input_file = 'assets/ref.png'
output_file = 'assets/ref_gpu.png'

# 1. Load an image from disk (on CPU)
image = cv2.imread(input_file, cv2.IMREAD_COLOR)
if image is None:
    print(f"Error: Failed to load the image file: {input_file}")
    print("Please make sure 'assets/ref.png' exists.")
    exit()

# 2. Upload the image to GPU memory
gpu_image = cv2.cuda.GpuMat()
gpu_image.upload(image)
print(f"Image uploaded to GPU: {gpu_image.size()} resolution")

# 3. Create a Gaussian filter and apply it to the image using the GPU
# Parameters: srcType, dstType, ksize (kernel size), sigmaX, sigmaY
gaussian_filter = cv2.cuda.createGaussianFilter(gpu_image.type(), -1, (15, 15), 0, 0)
gpu_blurred_image = gaussian_filter.apply(gpu_image)
print("Gaussian blur applied on GPU")

# 4. Download the result back to the CPU
blurred_image = gpu_blurred_image.download()
print("Result downloaded to CPU")

# 5. Save the result as a new file
cv2.imwrite(output_file, blurred_image)
print(f"Blurred image saved to: {output_file}")
