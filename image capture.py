import cv2 as cv
import numpy as np


def busornot(src):
    img = cv.imread(src)
    print(img.shape)
    bus_cascade_src = 'bus_front.xml'
    bus_cascade = cv.CascadeClassifier(bus_cascade_src)
    greys = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    bus = bus_cascade.detectMultiScale(greys, 1.1, 1)
    print(bus)
    cnt = 0
    for (x,y,w,h) in bus:
        cv.rectangle(img, (x,y), (x+w, y+h), (255,0,0), thickness=2)
        cnt += 1
    if w <= img.shape[1]//2 or h <= img.shape[0]//2:
        return False
    else:
        return True
    print(cnt, "bus is found")
    cv.imshow('bus', img)
    cv.waitKey(0)


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

path = r'C:/testing stage/SBS Scania front.jpg'
print(busornot(path))



