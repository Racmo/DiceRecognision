from ShapeDetector import ShapeDetector
import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture(0)

#wczytanie tla
_, background_image = cap.read()
background_image = cv2.cvtColor(background_image, cv2.COLOR_BGR2GRAY)

background_image = cv2.medianBlur(background_image, 5)
background_image = cv2.GaussianBlur(background_image, (5,5), 0)

#glowna petla
while(True):
    _, original_frame = cap.read()
    frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2GRAY)

    frame = cv2.medianBlur(frame, 5)
    frame = cv2.GaussianBlur(frame, (5,5), 0)

    cv2.absdiff(background_image, frame, frame)

    frame = cv2.threshold(frame, 100, 255, cv2.THRESH_BINARY)[1]

    #wyszukiwanie konturow
    contours = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    shape_detector = ShapeDetector()

    mask = np.ones(frame.shape[:2], dtype="uint8") * 255

    for contour in contours:
        if shape_detector.detect(contour) == 'square':
            cv2.drawContours(original_frame, [contour], -1, (0, 255, 0), 5)
            # area = cv2.boundingRect(np.zeros(contour))
            cv2.drawContours(mask, [contour], -1, 0, -1)

    mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV)[1]
    cropped_frame = cv2.bitwise_and(original_frame, original_frame, mask=mask)
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    cropped_frame = cv2.threshold(cropped_frame, 210, 255, cv2.THRESH_BINARY)[1]

    #wyswietlanie
    cv2.imshow('Edited', frame)
    cv2.imshow('Main', original_frame)
    cv2.imshow('Cropped', cropped_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#sprzatanie
cap.release()
cv2.destroyAllWindows()