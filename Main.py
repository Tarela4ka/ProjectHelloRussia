import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

cam = cv.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path=r'C:\Users\user\Downloads\gesture_recognizer.task')
options = vision.GestureRecognizerOptions(base_options=base_options)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

while True:
    success, frame = cam.read()
    if not success:
        print("-camera")
        break;
        
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize(image)
    if recognition_result.gestures:
        print(recognition_result.gestures[0][0].category_name)
        
    
    cv.imshow("Camera", frame)
    key = cv.waitKey(30) 
    if (key != -1):
        if chr(key) == 'p':
            pause = not pause;
            print(chr(key) + " pressed for pause")
        else:
            print(chr(key) + " pressed for exit")
            break;
        
cam.release()
cv.destroyAllWindows()
cv.waitKey(10)
