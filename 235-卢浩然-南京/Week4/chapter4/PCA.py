import cv2
import numpy as np


# 1. 对原始数据零均值化（中心化），
# 2. 求协方差矩阵，
# 3. 对协方差矩阵求特征向量和特征值，这些特征向量组成了新的特征空间。
class PCA:

    # 求均值
    def centralization(array):
        return np.array([np.mean(attr) for attr in array.T])

    # 中心化矩阵
    def getCentralizationMatrix(array):
        mean = PCA.centralization(array)
        print("均值:" , mean)
        return array - mean

    # 协方差矩阵
    def getCov(array):
        centerMatrix = PCA.getCentralizationMatrix(array)
        length = len(centerMatrix)
        print("中心矩阵:" , centerMatrix)
        return np.dot(centerMatrix.T , centerMatrix) / (length - 1)

    # 获取特征值
    def getEigenValue(array):
        # length = len(array)
        # identityMatrix = PCA.initIdentityMatrix(length)
        cov = PCA.getCov(array)
        print("协方差矩阵:" , cov)
        a, b = np.linalg.eig(cov)
        print("特征值:" , a)
        print("特征值向量" , b)
        return a , b

    def getEigenArray(array , size):
        value , eArray = PCA.getEigenValue(array)
        length = len(np.sort(value))
        # 取前几行(自定义)
        newArray = np.array([eArray[:,i] for i in range(size)])
        return newArray.T

    # 获取降维矩阵
    def dimensionalityReductionMatrix(array , size):
        eArray = PCA.getEigenArray(array , size)
        print("特征矩阵" , eArray)
        return np.dot(array , eArray)

    # # 创建单位矩阵
    # def initIdentityMatrix(length):
    #     array = [[0] * length] * length
    #     for i in range(length):
    #         for j in range(length):
    #             if(i == j):
    #                 array[i][j] = 1
    #             else:
    #                 continue
    #     return array

    def getImageArray(filePath):
        image = cv2.imread(filePath, 1)
        # 长，宽，通道
        originArray = image.shape[1]


    def mainFunction(filePath , size):
        # imageArray = PCA.getImageArray(filePath)
        imageArray = np.array([[10, 15, 29],
                  [15, 46, 13],
                  [23, 21, 30],
                  [11, 9,  35],
                  [42, 45, 11],
                  [9,  48, 5],
                  [11, 21, 14],
                  [8,  5,  15],
                  [11, 12, 21],
                  [21, 20, 25]])
        finalArray = PCA.dimensionalityReductionMatrix(imageArray , size)
        print("降维矩阵:" , finalArray)

PCA.mainFunction("111" , 2)