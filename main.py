from ShapeDetector import ShapeDetector
import cv2
import numpy as np
import imutils
import time

cap = cv2.VideoCapture(1)

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
    # frame = cv2.Canny(frame, 80,80)


    frame = cv2.threshold(frame, 50, 255, cv2.THRESH_BINARY)[1]

    #wyszukiwanie konturow
    contours = cv2.findContours(frame.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if imutils.is_cv2() else contours[1]

    shape_detector = ShapeDetector()

    mask = np.ones(frame.shape[:2], dtype="uint8") * 255

    for contour in contours:
        if shape_detector.detect(contour) == 'square':
            cv2.drawContours(original_frame, [contour], -1, (0, 255, 0), 2)
            # area = cv2.boundingRect(np.zeros(contour))
            cv2.drawContours(mask, [contour], -1, 0, -1) #dodanie do maski konturu kostki

    mask = cv2.threshold(mask, 100, 255, cv2.THRESH_BINARY_INV)[1]
    cropped_frame = cv2.bitwise_and(original_frame, original_frame, mask=mask) #wyciecie kostki z oryginalnego obrazu
    cropped_frame = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2GRAY)
    cropped_frame = cv2.threshold(cropped_frame, 210, 255, cv2.THRESH_BINARY)[1]

    kernel = np.ones((5, 5), np.uint8)

    h, w = cropped_frame.shape[:2]
    mask = np.zeros((h+2, w+2), np.uint8)

    #zamalowanie tla na bialo
    cv2.floodFill(cropped_frame, mask, (0,0), 255)

    #usuniecie 'ramki' wokol kostki
    cropped_frame = cv2.medianBlur(cropped_frame, 5)
    cropped_frame = cv2.GaussianBlur(cropped_frame, (3, 3), 0)
    cropped_frame = cv2.threshold(cropped_frame, 230, 255, cv2.THRESH_BINARY)[1]

    #Utworzenie blob detectora
    blob_detector_params = cv2.SimpleBlobDetector_Params()
    blob_detector_params.minCircularity = 0.5 #stopien zaokraglenia
    blob_detector = cv2.SimpleBlobDetector_create(blob_detector_params)

    keypoints = blob_detector.detect(cropped_frame)

    #zaznaczenie oczek na kostce
    text = 'Liczba oczek: ' + str(len(keypoints))
    original_frame = cv2.drawKeypoints(original_frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    cv2.putText(original_frame, text, (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

    #wyswietlanie
    # cv2.imshow('Edited', frame)
    cv2.imshow('Main', original_frame)
    # cv2.imshow('Cropped', cropped_frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#sprzatanie
cap.release()
cv2.destroyAllWindows()