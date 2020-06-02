import pandas as pd

import os
import math
import random

import sys
import cv2
import math
import numpy as np
import random
from matplotlib import pyplot as plt

def noise(img,snr):
    h=img.shape[0]
    w=img.shape[1]
    img1=img.copy()
    sp=h*w   # 计算图像像素点个数
    NP=int(sp*(1-snr))   # 计算图像椒盐噪声点个数
    for i in range (NP):
        randx=np.random.randint(1,h-1)   # 生成一个 1 至 h-1 之间的随机整数
        randy=np.random.randint(1,w-1)   # 生成一个 1 至 w-1 之间的随机整数
        if np.random.random()<=0.5:   # np.random.random()生成一个 0 至 1 之间的浮点数
            if img1.ndim == 2:
                img1[randx,randy]=0
            else:
                img1[randx, randy, 0] = 0
                img1[randx, randy, 1] = 0
                img1[randx, randy, 2] = 0
        else:
            if img1.ndim == 2:
                img1[randx, randy] = 255
            else:
                img1[randx, randy, 0] = 255
                img1[randx, randy, 1] = 255
                img1[randx, randy, 2] = 255
    return img1


# def sp_noise(image,prob):
#     '''
#     添加椒盐噪声
#     prob:噪声比例
#     '''
#     output = np.zeros(image.shape,np.uint8)
#     thres = 1 - prob
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             rdn = random.random()
#             if rdn < prob:
#                 output[i][j] = 0
#             elif rdn > thres:
#                 output[i][j] = 255
#             else:
#                 output[i][j] = image[i][j]
#     return output

def sp_noise(image,prob):
    '''
    添加椒盐噪声
    prob:噪声比例
    '''
    prob = prob / 2
    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            rdn = random.random()
            if rdn < prob:
                if image.ndim == 2:
                    output[i][j] = 0
                else:
                    output[i][j][0] = 0
                    output[i][j][1] = 0
                    output[i][j][2] = 0

            elif rdn > thres:
                if image.ndim == 2:
                    output[i][j] = 255
                else:
                    output[i][j][0] = 255
                    output[i][j][1] = 255
                    output[i][j][2] = 255
            else:
                output[i][j] = image[i][j]
    return output


if __name__ == "__main__":
    # path = "D:/ean13/BarcodeImage"
    # filetype = "txt"
    # img = cv2.imread('12.png')
    #
    # grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # #
    # grayImg = sp_noise(grayImg, 0.01)
    # # grayImg = noise(grayImg, 0.001)
    # median = cv2.medianBlur(grayImg, 3)
    # mean = cv2.blur(grayImg, (3,3))
    #
    # cv2.imshow("grayImg", grayImg)
    # cv2.imshow("median", median)
    # cv2.imshow("mean", mean)
    #
    # # cv2.imwrite("grayImg03.png", grayImg)
    # # cv2.imwrite("median03.png", median)
    # # cv2.imwrite("mean03.png", mean)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # path = "D:/ean13/BarcodeImage_hough45_addNoise01_addBlack2"
    path = "D:/ean13/BarcodeImage_hough45_addNoise03"
    filelist = os.listdir(path)
    sum = 0
    success = 0
    i = 0


    for i in range(1000):
        # r = random.randint(1, 1000)
        imagePath = path + "/" + str(i + 1) + ".png"
        sum = sum + 1
        img = cv2.imread(imagePath)
        img2 = sp_noise(img, 0.3)


        cv2.imwrite(imagePath, img2)


