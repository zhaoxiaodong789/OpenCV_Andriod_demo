import cv2
from math import *
import numpy as np
import random
import os


def rotate(img, degree):
    # img = cv2.imread("plane.jpg")
    # img = Image.open(pic)
    # img = np.array(img)
    height, width = img.shape[:2]

    # degree = 90
    # 旋转后的尺寸
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))

    matRotation = cv2.getRotationMatrix2D((width / 2, height / 2), degree, 1)

    matRotation[0, 2] += (widthNew - width) / 2  # 重点在这步，目前不懂为什么加这步
    matRotation[1, 2] += (heightNew - height) / 2  # 重点在这步

    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))

    # cv2.imshow("img", img)
    # cv2.imshow("imgRotation", imgRotation)
    # cv2.waitKey(0)
    return imgRotation



if __name__ == '__main__':
    path = "D:/ean13/BarcodeImage_hough"
    filelist = os.listdir(path)
    sum = 0
    success = 0
    i = 0


    for i in range(500):
        r = random.randint(1, 1000)
        imagePath = path + "/" + str(r) + ".png"
        sum = sum + 1
        img = cv2.imread(imagePath)

        degree = 180 - random.randint(0, 360)

        img2 = rotate(img, degree)

        cv2.imwrite(imagePath, img2)