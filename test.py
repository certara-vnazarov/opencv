import numpy as np
import cv2


def detect_objects(input_image):
    image = cv2.imread(input_image)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    #gray = cv2.medianBlur(gray, 5)
    #cv2.imwrite("gray.jpg", gray)
    edged = cv2.Canny(gray, 10, 250)
    #cv2.imwrite("edged.jpg", edged)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
    #cv2.imwrite("closed.jpg", closed)
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    total = 0
    for c in cnts:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            cv2.drawContours(image, [approx], -1, (0, 255, 0), 4)
            total += 1

    cv2.imwrite("output.jpg", image)


def face_regonition():
    face_cascade = cv2.CascadeClassifier(
        u'C:\Python27\Lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(
        u'C:\Python27\Lib\site-packages\cv2\data\haarcascade_eye.xml')

    url = 'https://www.sciencedaily.com/images/2017/09/170904120420_1_900x600.jpg'

    cap = cv2.VideoCapture(url)
    ret, img = cap.read()

    cv2.imshow('img', img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 3) #scale factor
    for (x, y, w, h) in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        eyes = eye_cascade.detectMultiScale(roi_gray)
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imwrite('new.jpg', img)


if __name__ == '__main__':
    #detect_objects("1.jpg")
    face_regonition()

