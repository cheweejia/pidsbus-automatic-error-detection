import cv2 as cv
import numpy as np


def is_bus(src):
    # img = cv.imread(src)
    img = src
    bus_cascade_src = 'bus_front.xml'
    bus_cascade = cv.CascadeClassifier(bus_cascade_src)
    greys = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bus = bus_cascade.detectMultiScale(greys, 1.1, 1, minSize=(50, 50))
    if len(bus) == 0:       #if detect nothing, reject the image
        return False
    frame = bus[:, 2:3]     #find box's length
    max_index = np.argmax(frame)       #find which box has the longest length
    (x,y,w,h) = bus[max_index]         #single out the largest box
    cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), thickness=2)     #lines 16 to 19 is for debugging
    # print("bus is found")
    # cv.imshow('bus', img)
    # cv.waitKey(0)
    if w <= img.shape[1]//4 or h <= img.shape[0]//4:    #if box too small, reject
        return False
    else:
        return True



# def webcamcapturedevice():
#     capture = cv.VideoCapture(0)
#     i = 0
#     while True:
#         isTrue, frame = capture.read()  # frame returns frame, isTrue returns whether the frame was successfully read
#         if isTrue:
#             cv.imshow('FRAME', frame)
#         if i % 30 == 0:
#             outcome = busornot(frame)
#             if outcome == True:
#
#         if cv.waitKey(20) & 0xFF == ord('d'):
#             break

# path = r'C:/bus front/130.jpg'
# print(is_bus(path))



