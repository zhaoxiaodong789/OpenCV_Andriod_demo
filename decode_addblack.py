import sys
import cv2
import math
import numpy as np
import os
from matplotlib import pyplot as plt
import pandas as pd
import random

DECODING_TABLE = {
    '0001101': 0, '0100111': 0, '1110010': 0,
    '0011001': 1, '0110011': 1, '1100110': 1,
    '0010011': 2, '0011011': 2, '1101100': 2,
    '0111101': 3, '0100001': 3, '1000010': 3,
    '0100011': 4, '0011101': 4, '1011100': 4,
    '0110001': 5, '0111001': 5, '1001110': 5,
    '0101111': 6, '0000101': 6, '1010000': 6,
    '0111011': 7, '0010001': 7, '1000100': 7,
    '0110111': 8, '0001001': 8, '1001000': 8,
    '0001011': 9, '0010111': 9, '1110100': 9,
}

EDGE_TABLE = {
    2: {2: 6, 3: 0, 4: 4, 5: 3},
    3: {2: 9, 3: '33', 4: '34', 5: 5},
    4: {2: 9, 3: '43', 4: '44', 5: 5},
    5: {2: 6, 3: 0, 4: 4, 5: 3},
}

FIRST_TABLE = {
    2: {2: 'O', 3: 'E', 4: 'O', 5: 'E'},
    3: {2: 'E', 3: 'O', 4: 'E', 5: 'O'},
    4: {2: 'O', 3: 'E', 4: 'O', 5: 'E'},
    5: {2: 'E', 3: 'O', 4: 'E', 5: 'O'},
}

FIRST_DETECT_TABLE = {
    'OOOOOO': 0, 'OOEOEE': 1, 'OOEEOE': 2, 'OOEEEO': 3, 'OEOOEE': 4,
    'OEEOOE': 5, 'OEEEOO': 6, 'OEOEOE': 7, 'OEOEEO': 8, 'OEEOEO': 9
}

INDEX_IN_WIDTH = (0, 4, 8, 12, 16, 20, 24, 33, 37, 41, 45, 49, 53)


def get_bar_space_width2(img, row):
    #row = img.shape[0] *1/2
    #row = int(img.shape[0] *1/2)
    currentPix = -1
    lastPix = -1
    pos = 0
    width = []
    # for i in range(img.shape[1]):#遍历一整行
    #     currentPix = img[row][i]
    #     if currentPix != lastPix:
    #         if lastPix == -1:
    #             lastPix = currentPix
    #             pos = i
    #         else:
    #             width.append( i - pos )
    #             pos = i
    #             lastPix = currentPix

    for i in range(img.shape[1]):#遍历一整行
        currentPix = img[row][i]
        if currentPix != lastPix:
            if lastPix == -1:
                lastPix = currentPix
                pos = i
            else:
                width.append( i - pos )
                pos = i
                lastPix = currentPix

    width.append(i - pos + 1)
    return width


def divide(t, l):
    if float(t) / l < 0.357:
        return 2
    elif float(t) / l < 0.500:
        return 3
    elif float(t) / l < 0.643:
        return 4
    else:
        return 5


def cal_similar_edge(data):
    similarEdge = []
    # 先判断起始符
    # limit = float(data[1] + data[2] + data[3]) / 3 * 1.5
    # if data[1] >= limit or data[2] >= limit or data[3] >= limit:
    #    return -1  # 宽度提取失败

    if float(data[2]/data[1]) > 0.5 and float(data[2]/data[1]) < 1.5 \
        and float(data[3] / data[2]) > 0.5 and float(data[3] / data[2]) < 1.5 \
        and float(data[58] / data[57]) > 0.5 and float(data[58] / data[57]) < 1.5 \
        and float(data[59] / data[58]) > 0.5 and float(data[59] / data[58]) < 1.5:


        index = 4
        while index < 54:
            # 跳过分隔符区间
            if index == 28 or index == 29 or index == 30 or index == 31 or index == 32:
                index += 1
                continue
            # 字符检测
            T1 = data[index] + data[index + 1]
            T2 = data[index + 1] + data[index + 2]
            L = data[index] + data[index + 1] + data[index + 2] + data[index + 3]
            similarEdge.append(divide(T1, L))
            similarEdge.append(divide(T2, L))
            index += 4

        return similarEdge

    return -1

def decode_similar_edge(edge):
    # 第一个字符一定是6，中国区
    first = []
    for i in range(0, 12, 2):
        first.append(FIRST_TABLE[edge[i]][edge[i + 1]])
    str = "".join(first)

    if str in FIRST_DETECT_TABLE.keys():
        barCode = [FIRST_DETECT_TABLE[str]]
    else:
        # barCode = [60]
        return -1

    for i in range(0, 24, 2):  # 每个字符两个相似边，共12个字符
        barCode.append(EDGE_TABLE[edge[i]][edge[i + 1]])
    return barCode


def decode_sharp(barCode, barSpaceWidth):
    for i in range(0, 13):
        if barCode[i] == '44':
            index = INDEX_IN_WIDTH[i]
            c3 = barSpaceWidth[index + 2]
            c4 = barSpaceWidth[index + 3]
            if c3 > c4:
                barCode[i] = 1
            else:
                barCode[i] = 7
        elif barCode[i] == '33':
            index = INDEX_IN_WIDTH[i]
            c1 = barSpaceWidth[index]
            c2 = barSpaceWidth[index + 1]
            if c1 > c2:
                barCode[i] = 2
            else:
                barCode[i] = 8
        elif barCode[i] == '34':
            index = INDEX_IN_WIDTH[i]
            c1 = barSpaceWidth[index]
            c2 = barSpaceWidth[index + 1]
            if c1 > c2:
                barCode[i] = 7
            else:
                barCode[i] = 1
        elif barCode[i] == '43':
            index = INDEX_IN_WIDTH[i]
            c2 = barSpaceWidth[index + 1]
            c3 = barSpaceWidth[index + 2]
            if c2 > c3:
                barCode[i] = 2
            else:
                barCode[i] = 8


def check_bar_code(barCode):
    evens = barCode[11] + barCode[9] + barCode[7] + barCode[5] + barCode[3] + barCode[1]
    odds = barCode[10] + barCode[8] + barCode[6] + barCode[4] + barCode[2] + barCode[0]
    sum = evens * 3 + odds
    if barCode[12] == (10 - sum % 10) % 10:
        return True
    else:
        return False


def ostu(grayImg):
    grayImg_width = grayImg.shape[1]
    grayImg_height = grayImg.shape[0]
    grayImg_1 = grayImg[0:int(grayImg_height/2), 0:int(grayImg_width/2)]
    grayImg_2 = grayImg[0:int(grayImg_height/2), int(grayImg_width/2):grayImg_width]
    grayImg_3 = grayImg[int(grayImg_height / 2):grayImg_height, 0:int(grayImg_width / 2)]
    grayImg_4 = grayImg[int(grayImg_height / 2):grayImg_height, int(grayImg_width/2):grayImg_width]

    ret, binaryImg_1 = cv2.threshold(grayImg_1, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)  # 二值化
    ret, binaryImg_2 = cv2.threshold(grayImg_2, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)  # 二值化
    ret, binaryImg_3 = cv2.threshold(grayImg_3, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)  # 二值化
    ret, binaryImg_4 = cv2.threshold(grayImg_4, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)  # 二值化

    binaryImg_new = grayImg.copy()

    binaryImg_new[0:int(grayImg_height/2), 0:int(grayImg_width/2)] = binaryImg_1
    binaryImg_new[0:int(grayImg_height/2), int(grayImg_width/2):grayImg_width] = binaryImg_2
    binaryImg_new[int(grayImg_height / 2):grayImg_height, 0:int(grayImg_width / 2)] = binaryImg_3
    binaryImg_new[int(grayImg_height / 2):grayImg_height, int(grayImg_width / 2):grayImg_width] = binaryImg_4


    return binaryImg_new





def barcode_detection(image):
    # image = cv2.imread(path)
    # image = cv2.imread('Foto(501).jpg')
    # image = cv2.imread('6.jpg')
    # image = cv2.imread('ean13.png')
    # image_copy = image.clone()
    # xuanzhuanImg = image.clone()

    imageGray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 转换成单通道图像

    # imageGaussian = cv2.GaussianBlur(imageGray, ksize=(3, 3), sigmaX=0)

    imageGaussian = imageGray

    imageX16S = cv2.Sobel(imageGaussian, cv2.CV_16S, 1, 0)
    imageY16S = cv2.Sobel(imageGaussian, cv2.CV_16S, 0, 1)
    imageSobelX = cv2.convertScaleAbs(imageX16S)
    imageSobelY = cv2.convertScaleAbs(imageY16S)

    imageSobelXY16S = imageX16S - imageY16S

    imageSobelOut = cv2.convertScaleAbs(imageSobelXY16S)


    Blurred = cv2.blur(imageSobelOut, (3, 3))



    ret, thresh = cv2.threshold(Blurred, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)


    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # erode = cv2.erode(closed, kernel=kernel_erode, iterations=3)
    #
    #
    # kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    # dilate = cv2.dilate(erode, kernel=kernel_dilate, iterations=5)

    kernel_erode = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    erode = cv2.erode(closed, kernel=kernel_erode, iterations=5)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 20))
    dilate = cv2.dilate(erode, kernel=kernel_dilate, iterations=5)



    # find the contours in the thresholded image
    _, contours, hierarchy = cv2.findContours(dilate, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)


    drawing = image.copy()
    # Approximate contours to polygons + get bounding rects and circles
    contours_poly = [None]*len(contours)
    boundRect = [None]*len(contours)
    minRect = [None] * len(contours)

    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])
        minRect[i] = cv2.minAreaRect(c)


    # find largest area's contour
    largest_area = 0
    largest_contour_index = 0
    for i in range(len(contours)):
        a = cv2.contourArea(contours[i], False)
        a = math.fabs(a)
        if (a > largest_area):
            largest_area = a
            largest_contour_index = i


    #因为数组会越界，发现出问题的都是列表为空，我就打算直接遇到空的就退出函数，2020.4.5
    if len(contours) == 0:
        return []



    # draw largest area
    color = (255, 0, 0)
    cv2.rectangle(drawing, (int(boundRect[largest_contour_index][0]), int(boundRect[largest_contour_index][1])), \
                  (int(boundRect[largest_contour_index][0] + boundRect[largest_contour_index][2]), int(boundRect[largest_contour_index][1] + boundRect[largest_contour_index][3])), \
                  color, 2)

    box = cv2.boxPoints(minRect[largest_contour_index])
    box1 = np.intp(box)  # np.intp: Integer used for indexing (same as C ssize_t; normally either int32 or int64)
    cv2.drawContours(drawing, [box1], 0, (0, 255, 0), 2)




    # 最小矩形的中心点、长宽和角度

    # 在C++中是一个结构体，在python中是一个list，内容为[center (x,y), (width, height), angle of rotation]
    angle = minRect[largest_contour_index][2]
    boxwidth = minRect[largest_contour_index][1][0]
    boxheight = minRect[largest_contour_index][1][1]
    # 定义旋转中心坐标
    center = minRect[largest_contour_index][0]

    # 旋转图片
    # 负数，顺时针旋转
    if 0 < abs(angle) and abs(angle) <= 45:
        angle = angle
    # 正数，逆时针旋转
    elif 45 < abs(angle) and abs(angle) <= 90:
        angle = 90 - abs(angle)

    angle0 = angle
    scale = 1

    # 获得旋转矩阵, 顺时针为负，逆时针为正
    roateM = cv2.getRotationMatrix2D(center, angle0, scale)

    image_copy = image.copy()
    xuanzhuanImg = image.copy()

    # 仿射变换
    xuanzhuanImg = cv2.warpAffine(image_copy, roateM, (xuanzhuanImg.shape[1], xuanzhuanImg.shape[0]))



    # 裁剪
    cut_width = cv2.boundingRect(contours[largest_contour_index])[2]
    cut_height = cv2.boundingRect(contours[largest_contour_index])[3]
    cut_X = cv2.boundingRect(contours[largest_contour_index])[0]
    cut_Y = cv2.boundingRect(contours[largest_contour_index])[1]

    xuanzhuanImg_cut = xuanzhuanImg[cut_Y:cut_Y+cut_height, cut_X:cut_X+cut_width]


    # 解码
    # barcode_decode(xuanzhuanImg_cut)
    img = xuanzhuanImg_cut.copy()
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成单通道图像

    # grayImg = cv2.medianBlur(grayImg, 3)  # 中值滤波
    # ret, grayImg = cv2.threshold(grayImg, 80, 255, cv2.THRESH_BINARY)  # 二值化

    # ret, grayImg = cv2.threshold(grayImg, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)  # 二值化

    grayImg = ostu(grayImg)

    for i in range(img.shape[0]):

        barCode = []
        barSpaceWidth = []
        similarEdge = []
        # 提取条空宽度
        barSpaceWidth = get_bar_space_width2(grayImg, i)

        if len(barSpaceWidth) >= 60:
            # 计算相似边数值
            similarEdge = cal_similar_edge(barSpaceWidth)

            if similarEdge == -1:
                continue

            # 相似边译码
            barCode = decode_similar_edge(similarEdge)
            if barCode == -1:
                continue

            # 针对‘#’译码
            decode_sharp(barCode, barSpaceWidth)
            # 校验
            valid = check_bar_code(barCode)
            if (valid == 1):
                break



    return barCode





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


def decode(img):

    # img = cv2.imread(path)
    grayImg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 转换成单通道图像
    grayImg = ostu(grayImg)

    for i in range(img.shape[0]):

        barCode = []
        barSpaceWidth = []
        similarEdge = []
        # 提取条空宽度
        barSpaceWidth = get_bar_space_width2(grayImg, i)

        if len(barSpaceWidth) >= 60:
            # 计算相似边数值
            similarEdge = cal_similar_edge(barSpaceWidth)

            if similarEdge == -1:
                continue

            # 相似边译码
            barCode = decode_similar_edge(similarEdge)
            if barCode == -1:
                continue

            # 针对‘#’译码
            decode_sharp(barCode, barSpaceWidth)
            # 校验
            valid = check_bar_code(barCode)
            if (valid == 1):
                break

    return barCode


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




def add_black(img):
    black_width = 200
    black_height = 100
    black = np.ones((black_height, black_width, 3))
    h_start = np.random.randint(0, 100)
    w_start = np.random.randint(0, 100)
    img[h_start:h_start+black_height, w_start:w_start+black_width] = black
    return img



if __name__ == '__main__':
    # 今天读取1000张代码，出现数组越界 2020.4.5
    # path = "D:/ean13/BarcodeImage"
    # path = "D:/ean13/BarcodeImage_hough45"
    path = "D:/ean13/BarcodeImage_hough45_addBlack3"
    # path = "D:/ean13/BarcodeImage_hough45_addNoise01"
    # path = "D:/ean13/BarcodeImage_hough45_addNoise01_addBlack1"
    # path = "D:/ean13/BarcodeImage_hough45_addNoise01_addBlack2"

    # path = "D:/ean13/BarcodeImage_hough45_addNoise03"
    filelist = os.listdir(path)
    sum = 0
    success = 0
    i = 0
    excelBarcode = pd.read_excel("ean13.xlsx", header = None)


    for i in range(0,1000):
        imagePath = path + "/" + str(i+1) + ".png"
        sum = sum + 1
        img = cv2.imread(imagePath)


        # 加噪声 测试鲁棒性
        # img = sp_noise(img, 0.4)
        # cv2.imwrite("D:/ean13/BarcodeImage_hough45_addNoise03_1"+ "/" + str(i+1) + ".png", img)
        img = cv2.medianBlur(img, 3)

        # 加黑块

        # 定位+解码
        barCode = barcode_detection(img)

        # 纯解码
        # barCode = decode(img)


        #excel 里面的数据如果是以0开头，会少了一位，在这里就在前面补充0
        excelBarcode_string = str(excelBarcode.iloc[i, 0])
        if len(excelBarcode_string) == 12:
            excelBarcode_string = "0" + excelBarcode_string

        barCode_string = ""

        if barCode == -1:
            continue

        for item in barCode:
            barCode_string = barCode_string + str(item)

        if barCode_string == excelBarcode_string:
            success = success + 1


        # print(i)
        # print(barCode_string)
        # print(excelBarcode_string)

    cv2.imshow("ss",img)
    print("sum of barcode: ", sum)
    print("sum of success: ", success)
    print("percentage of success: ", success / sum)





