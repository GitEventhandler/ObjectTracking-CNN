import gui.extra_function.structs as structs


def is_cross(p0: structs.Point, p1: structs.Point, p2: structs.Point):
    x1 = p1.x - p0.x
    y1 = p1.y - p0.y
    x2 = p2.x - p0.x
    y2 = p2.y - p0.y
    return x1 * y2 - x2 * y1


def is_line_line_intersection(line0: structs.Line, line1: structs.Line):
    p0 = line0.a_point
    p1 = line0.b_point
    p2 = line1.a_point
    p3 = line1.b_point
    result = False
    if (
            max(p0.x, p1.x) >= min(p2.x, p3.x)
            and max(p2.x, p3.x) >= min(p0.x, p1.x)
            and max(p0.y, p1.y) >= min(p2.y, p3.y)
            and max(p2.y, p3.y) >= min(p0.y, p1.y)
    ):
        if (is_cross(p0, p1, p2) * is_cross(p0, p1, p3) <= 0 and is_cross(p2, p3, p0) * is_cross(p2, p3, p1) <= 0):
            result = True

    return result


def is_point_in_rect(point: structs.Point, rect: structs.Rectangle):
    minx, miny = rect.a_point.to_cv_point()
    maxx, maxy = rect.c_point.to_cv_point()
    return (point.x >= minx and point.x <= maxx and point.y >= miny and point.y <= maxy)


def is_line_rect_intersection(line: structs.Line, rect: structs.Rectangle):
    if is_point_in_rect(line.a_point, rect) and is_point_in_rect(line.b_point, rect):
        return True
    pa, pb = rect.a_point, rect.c_point
    line0 = structs.Line()
    line0.a_point = rect.a_point
    line0.b_point = rect.d_point

    line1 = structs.Line()
    line1.a_point = rect.a_point
    line1.b_point = rect.b_point

    line2 = structs.Line()
    line2.a_point = rect.b_point
    line2.b_point = rect.c_point

    line3 = structs.Line()
    line3.a_point = rect.c_point
    line3.b_point = rect.d_point

    if is_line_line_intersection(line, line0):
        return True
    if is_line_line_intersection(line, line1):
        return True
    if is_line_line_intersection(line, line2):
        return True
    if is_line_line_intersection(line, line3):
        return True
    return False


def is_rect_rect_intersection(rect0: structs.Rectangle, rect1: structs.Rectangle):
    if is_point_in_rect(rect0.a_point, rect1):
        return True
    if is_point_in_rect(rect0.b_point, rect1):
        return True
    if is_point_in_rect(rect0.c_point, rect1):
        return True
    if is_point_in_rect(rect0.d_point, rect1):
        return True
    return False
