import cv2
import matplotlib.pyplot as plt
from bus_detection import detect_bus
from license_plate_recognition import anpr
from error_detection import detect_error

# Only works for images containing ONE bus
# Can be scaled to identify errors from images with multiple buses (it's a matter of writing the program)
def main():
    # Display image
    img = cv2.imread('data/9.jpg')
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

    # Detect bus from image
    detections = detect_bus(img)

    # Run analysis on the objects detected to determine if there were erroneous EDS
    eds_erroneous = detect_error(detections)
    if eds_erroneous:
        plate_number = anpr(img, detections)
        print("Bus " + plate_number + "'s EDS is erroneous.\n")

main()