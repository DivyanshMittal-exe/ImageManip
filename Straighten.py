import cv2
import mediapipe as mp
import os
from PIL import Image
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose




BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")


def calculate_angle(a, b, c):
    a = np.array(a)  # First
    b = np.array(b)  # Mid
    c = np.array(c)  # End

    radians = np.arctan2(c[1] - b[1], c[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    if 2*a[1] < b[1] + c[1]:
        angle = 180 + angle

    return 180+angle



with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for file in os.listdir(inputPath):
        print(file)
        img_initial = cv2.imread(inputPath + "\\" + file)
        print(img_initial)
        # cv2.cvtColor(img_initial, cv2.COLOR_BGR2GRAY)
        image = cv2.cvtColor(img_initial, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False


        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)


        landmarks = results.pose_landmarks.landmark

        # print(len(landmarks))
        LeftShoulder = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value]
        RightShoulder = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]
        Nose = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value]

        print(LeftShoulder)

        AngleOut = calculate_angle([Nose.x,Nose.y], [LeftShoulder.x,LeftShoulder.y], [RightShoulder.x,RightShoulder.y])
        print(AngleOut)

        rows,cols,ht = image.shape

        matrix = cv2.getRotationMatrix2D((rows/2,cols/2) ,AngleOut,1)

        outImage = cv2.warpAffine(image,matrix,(rows,cols))

        # cv2.imshow("out",outImage)

        # cv2.imwrite("Output1.png", outImage)
        if not cv2.imwrite(os.path.join(outputPath + "\\"+ file ), outImage):
            raise Exception("Could not write image")

        # time.sleep(10)







# cap = cv2.VideoCapture(0)
# ## Setup mediapipe instance
# with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
#     while cap.isOpened():
#         ret, frame = cap.read()
#
#         # Recolor image to RGB
#         image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         image.flags.writeable = False
#
#         # Make detection
#         results = pose.process(image)
#
#         # Recolor back to BGR
#         image.flags.writeable = True
#         image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
#
#         # Render detections
#         mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
#                                   mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
#                                   mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
#                                   )
#
#         cv2.imshow('Mediapipe Feed', image)
#
#         if cv2.waitKey(10) & 0xFF == ord('q'):
#             break
#
#     cap.release()
#     cv2.destroyAllWindows()