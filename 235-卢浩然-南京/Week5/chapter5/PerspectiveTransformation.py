import numpy as np
import cv2
import matplotlib.pyplot as plt

class PerspectiveTransformation:

    def get2Dposition(self, warpMartix, x, y):
        _3dx = x * warpMartix[0][0] + y * warpMartix[0][1] + warpMartix[0][2]
        _3dy = x * warpMartix[1][0] + y * warpMartix[1][1] + warpMartix[1][2]
        _3dz = x * warpMartix[2][0] + y * warpMartix[2][1] + warpMartix[2][2]
        tx = _3dx / _3dz
        ty = _3dy / _3dz
        return int(tx), int(ty)

    def getWarpMatrix(self, originArray, targetArray):
        line , row = targetArray.shape

        # 转化为一位数组(用于赋值)
        tempOrigin = originArray.flatten()
        tempTargetArray = targetArray.flatten()

        # 算数矩阵
        staticFunctionMatrix = np.zeros([2*line, 8])

        # 结果矩阵
        resultMatrix = np.mat(tempTargetArray).T

        for i in range(line * 2):
            if i % 2 == 0:
                staticFunctionMatrix[i][0] = tempOrigin[i]
                staticFunctionMatrix[i][1] = tempOrigin[i + 1]

                staticFunctionMatrix[i][2] = 1
                staticFunctionMatrix[i][3] = 0
                staticFunctionMatrix[i][4] = 0
                staticFunctionMatrix[i][5] = 0

                staticFunctionMatrix[i][6] = -1 * tempOrigin[i] * tempTargetArray[i]
                staticFunctionMatrix[i][7] = -1 * tempOrigin[i + 1] * tempTargetArray[i + 1]
            else:
                staticFunctionMatrix[i][3] = tempOrigin[i - 1]
                staticFunctionMatrix[i][4] = tempOrigin[i]

                staticFunctionMatrix[i][0] = 0
                staticFunctionMatrix[i][1] = 0
                staticFunctionMatrix[i][2] = 0
                staticFunctionMatrix[i][5] = 1

                staticFunctionMatrix[i][6] = -1 * tempOrigin[i - 1] * tempTargetArray[i - 1]
                staticFunctionMatrix[i][7] = -1 * tempOrigin[i] * tempTargetArray[i]

        staticFunctionMatrix = np.mat(staticFunctionMatrix)

        wrapMatrix = staticFunctionMatrix.I * resultMatrix
        wrapMatrix = np.array(wrapMatrix).T[0]
        wrapMatrix = np.insert(wrapMatrix, wrapMatrix.shape[0], 1.0, 0)  # 插入a_33 = 1
        wrapMatrix = wrapMatrix.reshape((3, 3))
        return wrapMatrix


    def initFlagPoint(self):
        originArray = np.array([[10.0, 457.0], [395.0, 291.0], [624.0, 291.0], [1000.0, 457.0]])
        targetArray = np.array([[46.0, 920.0], [46.0, 100.0], [600.0, 100.0], [600.0, 920.0]])
        wrapMatrix = PerspectiveTransformation.getWarpMatrix(self , originArray , targetArray)
        print("wrapMatrix:" , wrapMatrix)
        PerspectiveTransformation.transImage(self , wrapMatrix)

    def transImage(self , wrapMatrix):
        image = cv2.imread("image/aaa.jpg")
        tempImage = np.zeros_like(image)
        width, height , channel = image.shape
        for i in range(width):
            for j in range(height):
                x , y = PerspectiveTransformation.get2Dposition(self , wrapMatrix , i , j)
                if(x >= width or x < 0 or y >= height or y < 0):
                    continue
                tempImage[x][y] = image[i][j]
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[1].imshow(tempImage)
        plt.show()

PerspectiveTransformation.initFlagPoint(PerspectiveTransformation)