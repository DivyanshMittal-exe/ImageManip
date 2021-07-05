import cv2
import cvzone
import os
from cvzone.SelfiSegmentationModule import SelfiSegmentation



segmentor = SelfiSegmentation(0)

def RemoveBackground(img,color):
    return segmentor.removeBG(img,color)



if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(BASE_DIR)

    inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
    outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")
    for file in os.listdir(inputPath):
        img_initial = cv2.imread(inputPath + "\\" + file)
        img_out = RemoveBackground(img_initial, (255, 255, 255))
        if not cv2.imwrite(os.path.join(outputPath + "\\" + file), img_out):
            raise Exception("Could not write image")


