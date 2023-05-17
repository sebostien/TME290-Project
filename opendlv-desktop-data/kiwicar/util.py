class Vec2:
    x: float
    y: float

    def __init__(self, x: float, y: float):
        self.x = x
        self.y = y

    def x_int(self) -> int:
        return int(self.x)

    def y_int(self) -> int:
        return int(self.y)

    def add(self, other):
        return Vec2(self.x + other.x, self.y + other.y)


class Region:
    mid: Vec2
    area: float

    def __init__(self, x: float, y: float, area: float):
        self.mid = Vec2(x, y)
        self.area = area


