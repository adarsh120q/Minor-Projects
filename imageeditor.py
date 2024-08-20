import cv2
import numpy as np

# Read the input image
input_image = cv2.imread("20230101_124338.jpg")

# Check if the image is loaded successfully
if input_image is None:
    print("Error: Unable to load the input image.")
    exit()

# Apply Gaussian blur filter
blurred_image = cv2.GaussianBlur(input_image, (5, 5), 0)

# Apply sharpening filter
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]])
sharpened_image = cv2.filter2D(input_image, -1, kernel_sharpening)

# Apply edge detection filter
edges = cv2.Canny(input_image, 100, 200)

# Display the original image and the filtered images
cv2.imshow("Original Image", input_image)
cv2.imshow("Blurred Image", blurred_image)
cv2.imshow("Sharpened Image", sharpened_image)
cv2.imshow("Edge Detection", edges)

# Wait for a key press and close all windows
cv2.waitKey(0)
cv2.destroyAllWindows()
