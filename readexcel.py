import pandas as pd

import os
import math
import random

if __name__ == '__main__':
    path = "D:/ean13/BarcodeImage"
    filetype = "txt"

    test = pd.read_excel("ean13.xlsx")
    s = []
    for i in range(5):
        s.append(test.iloc[i, 0])
    print(s)

