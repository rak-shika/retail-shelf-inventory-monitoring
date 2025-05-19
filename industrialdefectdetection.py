import cv2
import numpy as np

# Load the image
image = cv2.imread('cracks_1.jpg') 
if image is None:
    raise FileNotFoundError("Image not found. Make sure 'cracks_1.jpg' is in your directory.")

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Gaussian Blur to reduce noise
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# Edge detection using Canny
edges = cv2.Canny(blurred, 50, 150)

# Dilate the edges to make cracks more visible
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(edges, kernel, iterations=1)

# Find contours
contours, _ = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding boxes and label them
for contour in contours:
    area = cv2.contourArea(contour)
    if area > 100:  # Filter out noise
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(image, 'Crack', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (36, 255, 12), 2)

# Display the result
cv2.imshow("Detected Cracks", image)
cv2.waitKey(0)
cv2.destroyAllWindows()