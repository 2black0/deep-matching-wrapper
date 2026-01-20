import cv2

print(cv2.getBuildInformation())

print("OpenCV Version:", cv2.__version__)
print("\nCUDA enabled devices:", cv2.cuda.getCudaEnabledDeviceCount())

print("\nAvailable functions in cv2.cuda:")
cuda_attrs = [attr for attr in dir(cv2.cuda) if not attr.startswith('_')]
for attr in sorted(cuda_attrs):
    print(f"  {attr}")