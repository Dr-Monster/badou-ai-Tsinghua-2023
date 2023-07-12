import numpy as np


def lsm():
    data = np.array([
        [1,3],
        [8,16],
        [7,18],
        [6,17],
        [9,15],
        [12,30]
    ])

    n = data.shape[0]

    xTotal = 0
    yTotal = 0
    x2Total = 0
    xyTotal = 0

    for i in range(n):
        xTotal += data[i][0]
        yTotal += data[i][1]
        xyTotal += data[i][0] * data[i][1]
        x2Total += data[i][0] * data[i][0]

    k = (n * xyTotal - xTotal * yTotal)/(n * x2Total - xTotal * xTotal)
    b = (yTotal - k * xTotal) / n

    print("k:" , k)
    print("b:" , b)

lsm()