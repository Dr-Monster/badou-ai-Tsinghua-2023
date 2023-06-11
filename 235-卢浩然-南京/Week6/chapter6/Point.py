class Point:
    x = 0
    y = 0
    long = 0

    def __int__(self, x, y, long):
        self.x = x
        self.y = y
        self.long = long

    def getX(self):
        return self.x

    def setX(self, x):
        self.x = x

    def getY(self):
        return self.y

    def setY(self, y):
        self.y = y

    def getLong(self):
        return self.long

    def setLong(self, long):
        self.long = long