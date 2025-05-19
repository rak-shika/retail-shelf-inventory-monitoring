import cv2
import numpy as np

# Load the image
image = cv2.imread('shelf_image.png')
resized_image = cv2.resize(image, (800, 600))

# Define bounding box colors for each product
colors = {
    "Pringles": (0, 0, 255),  # Red
    "Coca-Cola": (244, 194, 194),  # Pink
    "Kellogg's Corn Flakes": (255, 0, 0),  # Blue
    "Lay's": (0, 255, 0),  # Green
    "Penne": (0, 255, 255),  # Yellow
    "Eleolys": (255, 175, 0),  # Sky blue
    "Empty Space": (255, 255, 255)  # White
}

# Correct bounding boxes for the products visible in the image
boxes = {
    "Pringles": [(17, 40, 155, 300)],
    "Coca-Cola": [(200, 40, 450, 300),
        (17, 340, 210, 580)],
    "Kellogg's Corn Flakes": [(460, 40, 650, 300)],
    "Lay's": [(660, 40, 790, 300)],
    "Penne": [(520, 340, 650, 580)],
    "Eleolys": [(660, 340, 790, 580)],
    "Empty Space": [(220, 340, 510, 580)] 
}

# Draw bounding boxes and labels on the image
product_counts = {product: len(box_list) for product, box_list in boxes.items()}

for product, box_list in boxes.items():
    for box in box_list:
        x1, y1, x2, y2 = box
        cv2.rectangle(resized_image, (x1, y1), (x2, y2), colors[product], 2)
        cv2.putText(resized_image, product, (x1, y1 -10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[product], 2)
        
# Create a black canvas for the output text
text_canvas = np.zeros((250, 800, 3), dtype=np.uint8)

# Display product counts dynamically on the canvas
y_offset = 30
for product, count in product_counts.items():
    text = f"{count} {product}" + (" bottles" if product == "Coca-Cola" else ("s" if count > 1 else ""))
    cv2.putText(text_canvas, text, (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 30

# Concatenate the image and text output vertically
output = np.concatenate((resized_image, text_canvas), axis=0)

# Display the output image
cv2.imshow('retail shelf.jpg', output)
cv2.waitKey(0)
cv2.destroyAllWindows()