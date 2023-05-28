import copy
import random

import cv2


# 给一副数字图像加上椒盐噪声的处理顺序：
# 1.指定信噪比 SNR（信号和噪声所占比例） ，其取值范围在[0, 1]之间
# 2.计算总像素数目 SP， 得到要加噪的像素数目 NP = SP * SNR
# 3.随机获取要加噪的每个像素位置P（i, j）
# 4.指定像素值为255或者0。
# 5.重复3, 4两个步骤完成所有NP个像素的加噪
class SaltPepperNoise:

    def copyImage(originImage):
        newImage = copy.deepcopy(originImage)
        return newImage

    def handleImage(filePath):
        image = cv2.imread(filePath, 1)
        # 长，宽，通道
        height, width, channels = image.shape
        # 新增一个空图片对象
        # emptyImage = np.zeros((height, width, channels), np.uint8)

        emptyImage = SaltPepperNoise.copyImage(image)

        pixelSum = height * width
        # 0~1之间的实数
        snr = random.random()
        handlePixelSum = int(pixelSum * snr)

        for i in range(handlePixelSum):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            z = random.randint(0, 2)
            if (x + y + z) / 2 == 0:
                emptyImage[x, y, z] = 0
            else:
                emptyImage[x, y, z] = 255
        # cv2.imshow('origin', image)
        # cv2.waitKey()
        return emptyImage
