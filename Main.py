import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe import solutions

cam = cv.VideoCapture(0)

base_options = python.BaseOptions(model_asset_path=r'C:\Users\user\Downloads\Gesture.task')
options = vision.GestureRecognizerOptions(base_options=base_options, num_hands = 2)
recognizer = vision.GestureRecognizer.create_from_options(options)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

FONT_SIZE = 1
FONT_THICKNESS = 2
GESTURE_TEXT_COLOR = (255, 255, 255)

previous = None

def draw_landmarks_on_image(rgb_image, detection_result):
    hand_landmarks_list = detection_result.hand_landmarks
    handedness_list = detection_result.handedness
    annotated_image = np.copy(rgb_image)

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        handedness = handedness_list[idx]

        hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        hand_landmarks_proto.landmark.extend([
          landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
        ])
        solutions.drawing_utils.draw_landmarks(
          annotated_image,
          hand_landmarks_proto,
          solutions.hands.HAND_CONNECTIONS,
          solutions.drawing_styles.get_default_hand_landmarks_style(),
          solutions.drawing_styles.get_default_hand_connections_style())

        height, width, _ = annotated_image.shape
        x_coordinates = [landmark.x for landmark in hand_landmarks]
        y_coordinates = [landmark.y for landmark in hand_landmarks]
        text_x = int(min(x_coordinates) * width)
        text_y = int(min(y_coordinates) * height) - 10

        cv.putText(annotated_image, f"{detection_result.gestures[0][0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, (0, 0, 0), FONT_THICKNESS+2, cv.LINE_AA)
        cv.putText(annotated_image, f"{detection_result.gestures[0][0].category_name}",
                    (text_x, text_y), cv.FONT_HERSHEY_DUPLEX,
                    FONT_SIZE, GESTURE_TEXT_COLOR, FONT_THICKNESS, cv.LINE_AA)

    return annotated_image

def drawAllLetters(img, needToDraw):
    if needToDraw:
        print("text to draw - ", needToDraw)
        x = 20
        y = round(img.shape[0] * 0.9)
        print(x,y)
        cv.putText(img, needToDraw, (x, y), cv.FONT_HERSHEY_DUPLEX,
                        1, (0, 0, 0), 4, cv.LINE_AA)
        cv.putText(img, needToDraw, (x, y), cv.FONT_HERSHEY_DUPLEX,
                        1, (255,255,255), 2, cv.LINE_AA)
    return img

text = ''

while True:
    success, frame = cam.read()
    if not success:
        print("-camera")
        break;
        
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
    recognition_result = recognizer.recognize(image)
    
    if recognition_result.gestures:
        currGesture = recognition_result.gestures[0][0].category_name
        print(currGesture)
        if currGesture == previous:
            amount_of_frames_with_one_gesture += 1
            if amount_of_frames_with_one_gesture >= 30:
                if currGesture == 'space':
                    text += ' '
                elif currGesture == 'delete':
                    text = text[:-1]
                else:
                    text += f"{currGesture}"
                amount_of_frames_with_one_gesture = 0
        else:
            previous = recognition_result.gestures[0][0].category_name
            amount_of_frames_with_one_gesture = 0
        
        drawAllLetters(frame, text)
        image_with_handlanmarks = draw_landmarks_on_image(frame, recognition_result)
        cv.imshow("axuy", image_with_handlanmarks)
    else:
        drawAllLetters(frame, text)
        cv.imshow("axuy", frame)
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
