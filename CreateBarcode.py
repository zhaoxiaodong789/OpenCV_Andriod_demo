import pandas as pd

import os
import math
import random

if __name__ == '__main__':
    path = "D:/ean13/BarcodeImage"
    filetype = "txt"

    filelist = os.listdir(path)
    s = []


    for i in range(1001):
        a1 = random.randint(0, 9)
        a2 = random.randint(0, 9)
        a3 = random.randint(0, 9)
        a4 = random.randint(0, 9)
        a5 = random.randint(0, 9)
        a6 = random.randint(0, 9)
        a7 = random.randint(0, 9)
        a8 = random.randint(0, 9)
        a9 = random.randint(0, 9)
        a10 = random.randint(0, 9)
        a11 = random.randint(0, 9)
        a12 = random.randint(0, 9)


        odd = a2 + a4 + a6 + a8 + a10 + a12
        odd = 3 * odd
        e = a1 + a3 + a5 + a7 + a9 + a11

        sum = odd + e
        a13 = (10 - sum % 10) % 10
        bar = str(a1) + str(a2) + str(a3) + str(a4) + str(a5) + str(a6) +str(a7)+str(a8)+str(a9)+str(a10)+str(a11)+str(a12)+str(a13)


        s.append(bar)




    # test = pd.DataFrame(data=s)
    # print(test)
    #
    # test1 = test.iloc[:, 0]
    # print(test1)
    # test1.to_excel("ean131.xlsx", index = False, header = False)


    test = pd.DataFrame(data=s)
    print(test)

    test1 = test.iloc[:, 0]
    print(test1)
    # test1.to_excel("EAN13.xlsx", index = False, header = False)