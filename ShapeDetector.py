import cv2

class ShapeDetector:

    def detect(self, contour):
        shape = 'unknown'
        perimeter = cv2.arcLength(contour, True) #liczy obwod ksztaltu
        vertices =  cv2.approxPolyDP(contour, 0.04 * perimeter, True) #drugi parametr oznacza dokladnosc przyblizenia, zazwyczaj od 1-5% obwodu

        #jesli kontur ma wiecej niz 10 wierzcholkow to uznajemy go za kolo
        if len(vertices) > 5 and len(vertices) < 8:
            shape = 'circle'
        elif len(vertices) == 4:
            shape = 'square'
        else:
            shape = 'undefined'

        return shape