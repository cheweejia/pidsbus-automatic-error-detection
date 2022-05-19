import cv2 as cv
import imutils
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def resize(image, scale=0.5) :
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)
    dimensions = (width, height)
    return cv.resize(image, dimensions, interpolation = cv.INTER_AREA)

# Only for the purpose of POC. During actual data collection, we will determine the optimal
# position to crop the photo for better analysis.
def crop(image, scale=0.5) :
    return image[int(image.shape[0]*0.6):image.shape[0], int(image.shape[1] * 0.7):image.shape[1]]

# Read image and process the image
img = cv.imread("data/13.jpg")
processed_img = resize(img, 1)
# plt.imshow(cv.cvtColor(processed_img, cv.COLOR_BGR2RGB))
# plt.show()
gray_resized_img = cv.cvtColor(processed_img, cv.COLOR_RGB2GRAY)

# Object detection without deep learning
# Remove noise
bfilter = cv.bilateralFilter(gray_resized_img, 11, 17, 17)
edged = cv.Canny(bfilter, 30, 200) # Canny algorithm used to detect edges
# plt.imshow(cv.cvtColor(edged, cv.COLOR_BGR2RGB))
# plt.show()

# Find contours
keypoints = cv.findContours(edged.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
contours = imutils.grab_contours(keypoints)
contours = sorted(contours, key=cv.contourArea, reverse=True)[:10]

location = None
for contour in contours:
    approx = cv.approxPolyDP(contour, 10, True)
    if len(approx) == 4:
        location = approx
        break

# Apply mask
mask = np.zeros(gray_resized_img.shape, np.uint8)
new_image = cv.drawContours(mask, [location], 0, 255, -1)
new_image = cv.bitwise_or(gray_resized_img, gray_resized_img, mask=mask)

# Extract license plate
(x, y) = np.where(mask == 255)
(x1, y1) = (np.min(x), np.min(y))
(x2, y2) = (np.max(x), np.max(y))
cropped_image = gray_resized_img[x1 : x2, y1 : y2]
cropped_image = resize(cropped_image, 4)
cropped_image = cv.adaptiveThreshold(cropped_image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 85, 5)
kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
cropped_image = cv.morphologyEx(cropped_image, cv.MORPH_CLOSE, kernel)
# cropped_image = cv.dilate(cropped_image,kernel,iterations = 1)
# cropped_image = cv.erode(cropped_image,kernel,iterations = 1)
# plt.imshow(cv.cvtColor(cropped_image, cv.COLOR_BGR2RGB))
# plt.show()

# OCR using PyTesseract
config = "--psm 6"
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
print("License plate number: " + pytesseract.image_to_string(cropped_image, config=config))