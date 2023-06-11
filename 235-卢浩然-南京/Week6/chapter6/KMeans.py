import math

import numpy as np


# k-means聚类算法的分析流程：
# 第一步，确定K值，即将数据集聚集成K个类簇或小组。
# 第二步，从数据集中随机选择K个数据点作为质心（Centroid）或数据中心。
# 第三步，分别计算每个点到每个质心之间的距离，并将每个点划分到离最近质心的小组。
# 第四步，当每个质心都聚集了一些点后，重新定义算法选出新的质心。（对于每个簇，计
# 算其均值，即得到新的k个质心点）
# 第五步，迭代执行第三步到第四步，直到迭代终止条件满足为止（聚类结果不再变化）


class KMeans:

    def getCenterPointList(self, k, groupList):
        centerPointList = []
        for i in range(k):
            points = np.array(groupList[i])
            #求数组的均值
            centerPointList.append(points.mean(0))
        return centerPointList

    def divideGroup(self, data, k, centerPointList):
        size = data.shape[0]
        # K-V
        groupList = dict()
        for i in range(k):
            groupList[i] = []
        for i in range(size):
            distanceList = []
            for j in range(0, k):
                x1 = centerPointList[j][0] - data[i][0]
                y1 = centerPointList[j][1] - data[i][1]
                long = math.sqrt(x1 * x1 + y1 * y1)
                distanceList.append(long)
            min = np.min(distanceList)
            index = distanceList.index(min)
            groupList[index].append(data[i])
        newCenterPointList = KMeans.getCenterPointList(self, k, groupList)
        # 判断质心数组是否相等
        if np.all(np.array(centerPointList) == np.array(newCenterPointList)):
            return groupList
        else:
            self.divideGroup(self, data, k, centerPointList)

    def initData(self):
        data = np.array(
            [
                [1, 1],
                [2, 2],
                [1, 30],
                [6, 17],
                [7, 3],
                [9, 1],
                [5, 6],
                [6, 11],
                [4, 8],
                [3, 5]
            ]
        )
        # k = data.shape[0]
        k = 2
        centerPointList = [[1, 30], [6, 17]]
        print(KMeans.divideGroup(self, data, k, centerPointList))


KMeans.initData(KMeans)
