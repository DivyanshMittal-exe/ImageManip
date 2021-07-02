import cv2

import cv2
import mediapipe as mp
import os
from PIL import Image
import time
import numpy as np
# mp_drawing = mp.solutions.drawing_utils
# mp_pose = mp.solutions.pose

def incircle(point,radius):
    if (point[0]-radius)**2 + (point[1]-radius)**2 <= radius**2:
        return True
    else:
        return False



BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")




for file in os.listdir(inputPath):
    print(file)
    img_initial = cv2.imread(inputPath + "\\" + file)

def ImageCircle(img_initial,final_radius):

    grayImg = cv2.cvtColor(img_initial,cv2.COLOR_BGR2GRAY)

    face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")

    faces = face_cascade.detectMultiScale(grayImg,scaleFactor=1.1,minNeighbors=5)

    #print(faces)
    if len(faces)>0 :
        face = faces[0]

        face_img_raw = img_initial[face[1]:face[1]+face[2],face[0]:face[0]+face[2]]


        face_img = cv2.cvtColor(face_img_raw,cv2.COLOR_BGR2BGRA)

        # cv2.imshow("f",face_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        radius = int(face[2]/2)
        # center = [radius,radius]
        #
        for rows in range(face_img.shape[0]):
            for cols in range(face_img.shape[1]):
                if not incircle([rows,cols],radius):
                    face_img[rows][cols][3] = 0

        face_img_final = cv2.resize(face_img,(final_radius,final_radius))

        return face_img_final
    else:
        blankim = np.zeros((final_radius,final_radius),dtype=np.uint8)
        return cv2.cvtColor(blankim,cv2.COLOR_GRAY2BGR)
    # if not cv2.imwrite(os.path.join(outputPath + "\\" + file), face_img):
    #     raise Exception("Could not write image")
    # cv2.imshow("f",face_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
    outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")

    for file in os.listdir(inputPath):
        print(file)
        img_initial = cv2.imread(inputPath + "\\" + file)
        outImg = ImageCircle(img_initial, 30)
        cv2.imshow("f",outImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()