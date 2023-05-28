# This is a sample Python script.
import cv2

from GuassNoise import GuassNoise
from SaltPepperNoise import SaltPepperNoise
from PCA import PCA


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

filePath = "image/bear2.jpg"

# 高斯
# outImage = GuassNoise.handleImage(filePath , 50 , 100)

# 椒盐
# outImage = SaltPepperNoise.handleImage(filePath)

# cv2.imshow('demo', outImage)
# cv2.waitKey()

# PCA
PCA.mainFunction(filePath)

