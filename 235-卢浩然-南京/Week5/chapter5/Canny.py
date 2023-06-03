import cv2
import matplotlib.pyplot as plt
import numpy as np
import random


# 1.对图像进行灰度化
# 2.对图像进行高斯滤波：
#   根据待滤波的像素点及其邻域点的灰度值按照一定的参数规则进行加权平均。这样可以有效滤去理想图像中叠加的高频噪声。
# 3.检测图像中的水平、垂直和对角边缘（如Prewitt，Sobel算子等）,计算梯度。
# 4.对梯度幅值进行非极大值抑制：通俗意义上是指寻找像素点局部最大值，将非极大值点所对应的灰度值置为0，这样可以剔除掉一大部分非边缘的点。
# 5.用双阈值算法检测和连接边缘
class Canny:
    # Prewitt算子
    kernelx = np.array(
        [
            [1, 1, 1],
            [0, 0, 0],
            [-1, -1, -1]
        ]
    )

    kernely = np.array(
        [
            [-1, 0, 1],
            [-1, 0, 1],
            [-1, 0, 1]
        ]
    )

    # 灰度化图像
    def grayscaleImage(self, filePath):
        imageGrayscaleArray = cv2.imread(filePath)
        gray_img = cv2.cvtColor(imageGrayscaleArray, cv2.COLOR_RGB2GRAY)
        return gray_img

    # 高斯滤波
    def gaussFilterImage(self, filePath):
        gaussFilterImage = cv2.GaussianBlur(self.grayscaleImage(self, filePath), (5, 5), 0, 0)
        return gaussFilterImage

    # 边缘检测&非极大值抑制
    def nonmaxLimit(self, filePath):
        image = self.gaussFilterImage(self, filePath)

        # Sobel
        x = cv2.Sobel(image, cv2.CV_16S, 1, 0)
        y = cv2.Sobel(image, cv2.CV_16S, 0, 1)

        # Prewitt
        # x = cv2.filter2D(image, -1, self.kernelx)
        # y = cv2.filter2D(image, -1, self.kernely)

        # 转换数据 并 合成
        absX = cv2.convertScaleAbs(x)  # 格式转换函数
        absY = cv2.convertScaleAbs(y)

        # 图像混合后的数组
        result = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)

        width, height = result.shape

        tempImage = np.zeros_like(image)
        g1 = 0
        g2 = 0
        g3 = 0
        g4 = 0
        for i in range(1, width - 1):
            for j in range(1, height - 1):
                # 梯度>1
                if (result[i][j] > 1):
                    g1 = image[i - 1][j - 1]
                    g2 = image[i][j - 1]
                    g3 = image[i][j + 1]
                    g4 = image[i + 1][j + 1]
                # 梯度<1
                if (result[i][j] < 1):
                    g1 = image[i - 1][j - 1]
                    g2 = image[i][j - 1]
                    g3 = image[i][j + 1]
                    g4 = image[i + 1][j + 1]
                # 梯度=1
                if (result[i][j] == 1):
                    g1 = g2 = image[i - 1][j - 1]
                    g3 = g4 = image[i + 1][j + 1]
                # 梯度=-1
                if (result[i][j] == 1):
                    g1 = g2 = image[i + 1][j - 1]
                    g3 = g4 = image[i - 1][j + 1]

                dp1, dp2 = self.getSubPixel(self, result[i][j], g1, g2, g3, g4)
                if image[i, j] == max(image[i, j], dp1, dp2):
                    tempImage[i][j] = image[i][j]

        # 基于梯度统计信息计算法实现自适应阈值
        MAX = image.max()
        MIN = image.min()
        MED = np.median(image)
        average = (MAX + MIN + MED) / 3
        sigma = 0.33
        # 低阈值
        lowThreshold = max(0, (1 - sigma) * average)
        # 高阈值
        highThreshold = min(255, (1 + sigma) * average)

        imageEdge = np.zeros_like(tempImage)

        for i in range(1, width - 1):
            for j in range(1, height - 1):
                if tempImage[i][j] >= highThreshold:
                    imageEdge[i][j] = 255
                elif tempImage[i][j] > lowThreshold:
                    # python切片 https://blog.csdn.net/Cai_Xu_Kun/article/details/114978189
                    around = tempImage[i - 1: i + 2, j - 1: j + 2]
                    if around.max() >= highThreshold:
                        imageEdge[i, j] = 255

        fig, axes = plt.subplots(1, 3)
        axes[0].imshow(image)
        axes[1].imshow(result)
        axes[2].imshow(imageEdge)
        plt.show()

    # 计算亚像素
    def getSubPixel(self, weight, g1, g2, g3, g4):
        dp1 = weight * g1 + (1 - weight) * g2
        dp2 = weight * g3 + (1 - weight) * g4
        return dp1, dp2


Canny.nonmaxLimit(Canny, "image/aaa.jpg")
