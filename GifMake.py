import cv2
from CropImage import incircle
from Backgound import RemoveBackground
import mediapipe as mp
import os
from PIL import Image
import time
import numpy as np
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

def ImageCircle(img,final_radius):
    rows,cols,c = img.shape
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        landmarks = results.pose_landmarks.landmark
        Nose = landmarks[mp_pose.PoseLandmark.NOSE.value]
        RowCenter = int(Nose.y * rows)
        ColCenter = int(Nose.x * cols)

        if cols < rows:
            radius = int(cols / 2)
            ColCenter = radius
            if RowCenter < radius:
                RowCenter = radius
            elif rows - RowCenter < radius:
                RowCenter = rows - radius
        else:
            radius = int(rows / 2)
            RowCenter = radius
            if ColCenter < radius:
                ColCenter = radius
            elif cols - ColCenter < radius:
                ColCenter = cols - radius

        face_img_raw = img[RowCenter - radius:RowCenter + radius, ColCenter - radius:ColCenter + radius]


        image = cv2.cvtColor(face_img_raw, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        landmarks = results.pose_landmarks.landmark
        LeftEye = landmarks[mp_pose.PoseLandmark.LEFT_EYE.value]
        RightEye = landmarks[mp_pose.PoseLandmark.RIGHT_EYE.value]

        LeftEyeRow = int(LeftEye.y * rows)
        LeftEyeCol = int(LeftEye.x * cols)
        RightEyeRow = int(RightEye.y * rows)
        RightEyeCol= int(RightEye.x * cols)

        NewLeftEyeRow = NewRightEyeRow = int(0.45*radius)
        NewLeftEyeCol = int(0.32*radius)
        NewRightEyeCol = int(0.68 * radius)
        slope = (RightEyeRow - LeftEyeRow)/(RightEyeCol-LeftEyeCol)
        slope = -1/slope
        CenterRow = (RightEyeRow+LeftEyeRow)/2
        CenterCol = (RightEyeCol+LeftEyeCol)/2
        dist = np.sqrt((RightEyeRow-LeftEyeRow)**2+(RightEyeCol-LeftEyeCol)**2)
        ThirdCenterCol =CenterCol + dist*np.cos(np.arctan(slope))
        ThirdCenterRow =CenterRow + dist*np.sin(np.arctan(slope))
        NewThirdCol = int(0.5 * radius)
        NewThirdRow = int(0.7617 * radius)

        # p1 = np.float32([[RightEyeRow,RightEyeCol],[LeftEyeRow,LeftEyeCol],[ThirdCenterRow,ThirdCenterCol]])
        # p2 = np.float32([[NewRightEyeRow,NewRightEyeCol],[NewLeftEyeRow,NewLeftEyeCol],[NewThirdRow,NewThirdCol]])

        p1 = np.float32([[RightEyeCol, RightEyeRow], [LeftEyeCol, LeftEyeRow], [ThirdCenterCol, ThirdCenterRow]])
        p2 = np.float32([[NewRightEyeCol, NewRightEyeRow], [NewLeftEyeCol, NewLeftEyeRow], [NewThirdCol, NewThirdRow]])

        matrix = cv2.getAffineTransform(p1,p2)
        final_img = cv2.warpAffine(face_img_raw,matrix,(2*radius,2*radius))

        return final_img


















if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
    outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")

    for file in os.listdir(inputPath):
        print(file)
        img_initial = cv2.imread(inputPath + "\\" + file)
        outImg = ImageCircle(img_initial, 30)
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), outImg):
            raise Exception("Could not write image")