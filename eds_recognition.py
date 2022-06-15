import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

def resize(image, scale=0.5):
    height = int(image.shape[0] * scale)
    width = int(image.shape[1] * scale)
    dimensions = (width, height)
    return cv2.resize(image, dimensions, interpolation = cv2.INTER_AREA)

def crop_eds(image, scale=0.5) :
    return image[int(image.shape[0] * 0.3):int(image.shape[0] * scale), int(image.shape[1]*0.4):image.shape[1]]

def extract_dest(image) :
    # image[height, width]
    return image[0:image.shape[0], 0:int(image.shape[1] * 0.5)]

def extract_serv_num(image) :
    # image[height, width]
    return image[0:image.shape[0], int(image.shape[1] * 0.5):image.shape[1]]

def eds_to_text(img) :
    # convert to grayscale
    small = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)

    # threshold the image
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    plt.imshow(cv2.cvtColor(bw, cv2.COLOR_BGR2RGB))
    plt.show()

    # Morphological transformation
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    # MORPH_CLOSE -> dilation followed by erosion
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    plt.imshow(cv2.cvtColor(connected, cv2.COLOR_BGR2RGB))
    plt.show()

    # Find contours
    # using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    mask = np.zeros(bw.shape, dtype=np.uint8)
    texts = []

    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)

        # Ratio of object to bounding rectangle
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)

        if r > 0.45 and w > 10 and h > 10:
            texts.append(resize(img.copy()[y:y+h+1, x:x+w+1], 1.5))
            cv2.rectangle(img, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2)
            # plt.imshow(cv2.cvtColor(rgb[y:y+h, x:x+w], cv2.COLOR_BGR2RGB))
            # plt.show()

    # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # plt.show()

    # OCR using PyTesseract
    # Assume a single uniform block of text.
    config = "--psm 6"
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    for text in texts:
        return pytesseract.image_to_string(text, config=config)

def read_eds(img) :
    eds = crop_eds(img)
    rgb = cv2.pyrDown(eds)
    dest = extract_dest(rgb)
    serv_num = extract_serv_num(rgb)
    return "Service number: \n" + eds_to_text(serv_num) + "\nInformation: \n" + eds_to_text(dest)