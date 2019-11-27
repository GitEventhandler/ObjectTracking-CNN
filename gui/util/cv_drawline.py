import numpy as np
import cv2

# 公有变量
_cv_draw_line_image = None
_cv_draw_line_image_copy = None
_cv_draw_line_drawing = False
_cv_draw_line_ix = 0
_cv_draw_line_iy = 0
_cv_draw_line_sx = 0
_cv_draw_line_sy = 0
_cv_draw_line_lines = []
_cv_draw_line_color = (255, 0, 0)


# 鼠标事件回调函数
def draw(event, x, y, flags, params):
    global _cv_draw_line_ix, _cv_draw_line_iy, _cv_draw_line_sx, _cv_draw_line_sy, _cv_draw_line_drawing, _cv_draw_line_image, _cv_draw_line_image_copy
    global _cv_draw_line_color
    _cv_draw_line_ix = x
    _cv_draw_line_iy = y
    # 左键按下
    if (event == cv2.EVENT_LBUTTONDOWN):
        # 还没开始，按下开始后记录初始点
        if _cv_draw_line_drawing == False:
            _cv_draw_line_drawing = True
            _cv_draw_line_sx = x
            _cv_draw_line_sy = y
    # 左键抬起
    if (event == cv2.EVENT_LBUTTONUP):
        if _cv_draw_line_drawing:
            _cv_draw_line_drawing = False
            _cv_draw_line_image_copy = np.copy(_cv_draw_line_image)
            cv2.line(_cv_draw_line_image_copy, pt1=(_cv_draw_line_ix, _cv_draw_line_iy),
                     pt2=(_cv_draw_line_sx, _cv_draw_line_sy), color=_cv_draw_line_color, thickness=3)
            # 划线成功，计入
            global _cv_draw_line_lines
            _cv_draw_line_lines.append([(_cv_draw_line_ix, _cv_draw_line_iy), (_cv_draw_line_sx, _cv_draw_line_sy)])
    # 鼠标移动时划线
    if (event == cv2.EVENT_MOUSEMOVE):
        if _cv_draw_line_drawing:
            _cv_draw_line_image_copy = np.copy(_cv_draw_line_image)
            cv2.line(_cv_draw_line_image_copy, pt1=(_cv_draw_line_ix, _cv_draw_line_iy),
                     pt2=(_cv_draw_line_sx, _cv_draw_line_sy), color=_cv_draw_line_color, thickness=3)


def selectLines(windTitle='Window', img = None):
    # 创建窗体
    global _cv_draw_line_drawing, _cv_draw_line_image, _cv_draw_line_image_copy, _cv_draw_line_lines
    cv2.namedWindow(windTitle)
    _cv_draw_line_image = np.copy(img)
    _cv_draw_line_image_copy = np.copy(_cv_draw_line_image)
    # 添加回调函数
    cv2.setMouseCallback(windTitle, draw)

    # 进入主循环
    while (True):

        cv2.imshow(windTitle, _cv_draw_line_image_copy)
        key = cv2.waitKey(20)
        if key == ord('c'):
            # 取消划线
            _cv_draw_line_drawing = False
        if key == 27:
            # 按下ESC
            break
    cv2.destroyWindow(windTitle)
    return _cv_draw_line_lines


'''
if __name__ == '__main__':
    line = selectLines(np.ones((512, 512, 3)))
    for l in line:
        print(l)

'''
