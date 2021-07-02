import numpy as np
import random
import os
import cv2
import CropImage

WIDTH = 1000
HEIGHT = 1000


TotalFalse = 0
TotalCircles = 300

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(BASE_DIR)

inputPath = os.path.join(BASE_DIR + "\\pythonProject\\Input")
outputPath = os.path.join(BASE_DIR + "\\pythonProject\\Output")

print("Hi")




def dist(a,b,c,d):
    return np.sqrt((a-b)**2+(c-d)**2)


def withinedge(x,y,radius):
    if x + radius<WIDTH and x > radius and y+radius<HEIGHT and y > radius:
        return True
    return False


class Circle:
    def __init__(self,x,y):
        self.x = x
        self.y= y
        self.r = 1
        self.growing = True
    def grow(self):
        if self.growing:
            global TotalFalse
            if not withinedge(self.x,self.y,self.r):
                TotalFalse += 1
                self.growing = False
            else:
                for circle in CircleList:
                    if circle.x != self.x and circle.y != self.y:
                        if dist(circle.x,self.x,circle.y,self.y) < circle.r + self.r :
                            self.growing = False
                            TotalFalse += 1

        if self.growing:
            self.r += 1





CircleList = []

print("Hi")

TemplateImg = cv2.imread(os.path.join(BASE_DIR + "\\pythonProject\\Bitmap.png"))
print(os.path.join(BASE_DIR + "\\pythonProject\\Bitmap.png"))
# BlankImg = cv2.cvtColor(BlankImg_raw,cv2.COLOR_GRAY2BGRA)


i = 0
while i < TotalCircles-1:
    x = random.randint(10, WIDTH - 10)
    y = random.randint(10, HEIGHT - 10)

    if TemplateImg[y][x][0] == 0:
        continue

    for circle in CircleList:
        if (circle.x - x)**2 + (circle.y-y)**2 < circle.r**2:
            break
    else:
        CircleList.append(Circle(x,y))
        i+=1
        # if  TotalFalse < TotalCircles:
        #     for circle in CircleList:
        #         if random.random() <=1:
        #             circle.grow()

anyo = 30

print( len(CircleList))
# CircleList.append(Circle(anyo,anyo))
# CircleList.append(Circle(WIDTH - anyo,anyo))
# CircleList.append(Circle(WIDTH -anyo,HEIGHT-anyo))
# CircleList.append(Circle(anyo,HEIGHT-anyo))
#
#
# for i in range(4,TotalCircles):
#     TwoCircles = random.sample(set(CircleList), 2)
#     newX = int((TwoCircles[0].x + TwoCircles[1].x)/2)
#     newy = int((TwoCircles[0].y + TwoCircles[1].y)/2)
#     CircleList.append(Circle(newX,newy))


print("Made Circle List")

while TotalFalse < len(CircleList):
    for circle in CircleList:
        if random.random() < 2:
            circle.grow()

print("Circles Grown")




BlankImg_raw = np.zeros((WIDTH,HEIGHT),dtype=np.uint8)
BlankImg = cv2.cvtColor(BlankImg_raw,cv2.COLOR_GRAY2BGRA)


for file in os.listdir(inputPath):
    print(file)
    img_initial = cv2.imread(inputPath + "\\" + file)

# grayImg = cv2.cvtColor(img_initial,cv2.COLOR_BGR2GRAY)
#
# face_cascade =  cv2.CascadeClassifier(cv2.data.haarcascades +"haarcascade_frontalface_default.xml")
#
# faces = face_cascade.detectMultiScale(grayImg,scaleFactor=1.1,minNeighbors=5)


for circle in CircleList:
    img_out = CropImage.ImageCircle(img_initial,2*circle.r)
    # BlankImg[circle.y-circle.r:circle.y+circle.r,circle.x-circle.r:circle.x+circle.r] = img_out
    for row in range(2*circle.r):
        for col in range(2*circle.r):
            if img_out[col][row][3]!=0:
                BlankImg[circle.y-circle.r + col][circle.x-circle.r+row] = img_out[col][row]

# cv2.imshow("f",BlankImg)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

if not cv2.imwrite(os.path.join(outputPath + "\\" + file), BlankImg):
        raise Exception("Could not write image")
