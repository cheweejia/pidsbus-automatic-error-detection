import cv2
import matplotlib.pyplot as plt
from image_capture_revised import is_bus
from eds_recognition import read_eds
from license_plate_recognition import anpr

def main():
    # while True:
    img = cv2.imread('data/8.jpg')
    if is_bus(img):
        print(read_eds(img))
        # if !is_valid_eds(eds):
        print(anpr(img))
            # send_report(img, eds)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
main()