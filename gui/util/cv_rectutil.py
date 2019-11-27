import cv2


# draw a rect with opencv-python
def draw_rect(img, rect, rgb=(0, 255, 0)):
    x, y, w, h = rect
    x = int(x)
    y = int(y)
    w = int(w)
    h = int(h)
    # RGB->BGR
    return cv2.rectangle(img, (x, y), (x + w, y + h), (rgb[2], rgb[1], rgb[0]), 2)


# draw a line with opencv-python
def draw_line(img, p0=(), p1=(), rgb=(0, 255, 0)):
    return cv2.line(img, p0, p1, (rgb[2], rgb[1], rgb[0]), 2, 4)


# draw lines, input [[(p0x,p0y),(p1x,p1y)],[...]]
def draw_lines(img, lines=[], rgb=(0, 255, 0)):
    for line in lines:
        img = draw_line(img, line[0], line[1], rgb)
    return img


# draw rects
def draw_rects(img, rects, rgb=(0, 255, 0)):
    for rect in rects:
        img = draw_rect(img, rect, rgb)
    return img
