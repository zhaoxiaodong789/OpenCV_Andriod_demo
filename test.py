import cv2
import os
import pandas as pd
import numpy as np


def add_black(img):
    # black_width = 200
    # black_height = 100
    black_width = np.random.randint(200, 400)
    black_height = np.random.randint(100, 150)
    black = np.ones((black_height, black_width, 3))
    h_start = np.random.randint(0, 100)
    w_start = np.random.randint(0, 100)
    img[h_start:h_start+black_height, w_start:w_start+black_width] = black
    return img

if __name__ == '__main__':
    # 今天读取1000张代码，出现数组越界 2020.4.5
    # path = "D:/ean13/BarcodeImage"
    path = "D:/ean13/BarcodeImage_hough45"
    filelist = os.listdir(path)
    sum = 0
    success = 0
    i = 0
    excelBarcode = pd.read_excel("ean13.xlsx", header = None)

    imagePath = path + "/" + str(801 + 1) + ".png"

    image = cv2.imread(imagePath)
    #
    black_width = 200
    black_height = 50
    black = np.ones((black_height, black_width, 3))
    #
    # print(black)

    # image[100:200, 100:300] = black
    h_start = 50
    w_start = 100
    image[h_start:h_start + black_height, w_start:w_start + black_width] = black

    # image = add_black(image)
    cv2.imshow("sf", image)

    print(image)
    print(image.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



