import cv2
import PyQt5.QtGui as QtGui
from PyQt5.QtGui import QImage


def cvimage_to_qpixmap(cv_image, qwidth, qheight):
    height, width, bytesPerComponent = cv_image.shape
    bytesPerLine = 3 * width
    cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB, cv_image)
    qimage = QImage(cv_image.data, width, height, bytesPerLine, QImage.Format_RGB888)
    qpixmap = QtGui.QPixmap.fromImage(qimage)
    qpixmap = qpixmap.scaled(qwidth, qheight)
    return qpixmap
