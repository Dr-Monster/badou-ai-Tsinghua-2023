import random
import cv2
import numpy as np
# from FileUtils import CustomFileUtils

# 给一副数字图像加上高斯噪声的处理顺序如下：
# a. 输入参数sigma 和 mean
# b. 生成高斯随机数
# d. 根据输入像素计算出输出像素
# e. 重新将像素值放缩在[0 ~ 255]之间
# f. 循环所有像素
# g. 输出图像

class GuassNoise:
    def getGuassNumber(pixelIn , avg , sigma):
        # 高斯随机数
        # avg:平均
        # sigma:标准偏差
        pixelOut = random.gauss(avg, sigma) + pixelIn
        if(pixelOut > 255):
            pixelOut = pixelOut * 255 / (255 + avg)
        return pixelOut

    def handleImage(filePath , avg , sigma):
        image = cv2.imread(filePath, 1)
        # 长，宽，通道
        height, width, channels = image.shape
        # 新增一个空图片对象
        emptyImage = np.zeros((height, width, channels), np.uint8)
        for i in range(width):
            for j in range(height):
                for k in range(channels):
                    emptyImage[i, j , k] = GuassNoise.getGuassNumber(image[i, j , k] , avg , sigma)

        cv2.imshow('origin', image)
        cv2.waitKey()
        return emptyImage
