class Point:
    def __init__(self):
        self.x = int()
        self.y = int()

    def from_cv_point(self, cv_point):
        self.x, self.y = list(cv_point)

    def to_cv_point(self):
        return (self.x, self.y)


'''
a_point *------* b_point
        |      |
        |      |
        |      |
d_point *------* c_point
'''


class Rectangle:
    def __init__(self):
        self.a_point = Point()
        self.b_point = Point()
        self.c_point = Point()
        self.d_point = Point()
        self.width = int()
        self.height = int()

    def center(self):
        p = Point()
        p.x = self.a_point.x + self.width / 2
        p.y = self.a_point.y + self.height / 2
        return p

    def from_cv_rect(self, cv_rect):
        x, y, self.width, self.height = cv_rect
        self.a_point.from_cv_point([x, y])
        self.b_point.from_cv_point([x + self.width, y])
        self.c_point.from_cv_point([x + self.width, y + self.height])
        self.d_point.from_cv_point([x, y + self.height])

    def to_cv_rect(self):
        return [self.a_point.x, self.a_point.y, self.width, self.height]


'''
a_point *------* b_point
'''


class Line:
    def __init__(self):
        self.a_point = Point()
        self.b_point = Point()

    def from_cv_line(self, line):
        self.a_point.from_cv_point(line[0])
        self.b_point.from_cv_point(line[1])

    def to_cv_line(self):
        return [self.a_point.to_cv_point(), self.a_point.to_cv_point()]


class RectProblem:
    def __init__(self):
        self.which_rect = Rectangle()
        self.line_touched = []
        self.rect_trespassed = []
