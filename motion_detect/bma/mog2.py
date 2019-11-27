import cv2
import numpy as np


class Mog2Detect:
    def __init__(self, min_roi_threshold=500):
        # MOG2 Background Subtractor
        self._bgs_mog2 = cv2.createBackgroundSubtractorMOG2(detectShadows=False, history=500, varThreshold=20)
        # ROI Threshold
        self._min_box_threshold = min_roi_threshold
        # This value means whether the init progress has been done yet
        self._is_inited = False

    def clear(self):
        self._is_inited = False

    # Fast Non Max Suppression results
    def fast_non_max_suppression(self, cvboxes, overlapThresh):
        boxes = np.array([[x, y, x + w, y + h] for (x, y, w, h) in cvboxes])
        # if there are no boxes, return an empty list
        if len(boxes) == 0:
            return []

        # if the bounding boxes integers, convert them to floats --
        # this is important since we'll be doing a bunch of divisions
        if boxes.dtype.kind == "i":
            boxes = boxes.astype("float")

        # initialize the list of picked indexes
        pick = []

        # grab the coordinates of the bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]

        # compute the area of the bounding boxes and sort the bounding
        # boxes by the bottom-right y-coordinate of the bounding box
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)

        # keep looping while some indexes still remain in the indexes
        # list
        while len(idxs) > 0:
            # grab the last index in the indexes list and add the
            # index value to the list of picked indexes
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])

            # compute the width and height of the bounding box
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)

            # compute the ratio of overlap
            overlap = (w * h) / area[idxs[:last]]

            # delete all indexes from the index list that have
            idxs = np.delete(idxs, np.concatenate(([last],
                                                   np.where(overlap > overlapThresh)[0])))

        # return only the bounding boxes that were picked using the
        # integer data type
        return [[x, y, m - x, n - y] for (x, y, m, n) in boxes[pick].astype("int").tolist()]

    # Get MOG2 Foreground feature, input : np.array
    def get_mog2_foreground(self, image):
        return self._bgs_mog2.apply(image)

    # Median filter
    def get_median_filted(self, image):
        # Avg fiter, core size = 9*9
        return cv2.medianBlur(image, 11)

    # A custom convolution filter which aim to fill the gaps
    def filter2d(self, image):
        kernel = np.array((
            [0, 1, 1, 1, 0],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [1, 0, 0, 0, 1],
            [0, 1, 1, 1, 0]
        ))
        return cv2.filter2D(image, -1, kernel)

    # Tracker Result filter
    def box_filter(self, boxes):
        boxes = self.fast_non_max_suppression(boxes, 0.65)
        return boxes

    # The train only option
    def train_only(self, image):
        self.get_mog2_foreground(image)

    # Get result from filters
    def get_object_boxes(self, image):
        if not self._is_inited:
            self._is_inited = True
            self.train_only(image)
            return []
        fmask = self.get_mog2_foreground(image)
        # Filtered dot nois
        fmask = self.get_median_filted(fmask)
        tmask = cv2.threshold(fmask.copy(), 200, 255, cv2.THRESH_BINARY)[1]
        eroded = cv2.erode(tmask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=3)
        dilated = cv2.dilate(eroded, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        # DEBUG USING
        # cv2.imshow('dilated', dilated)
        # Get raw result
        image, contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Initialize result lists
        all_boxes = [[0 for i in range(4)] for j in range(len(contours))]
        index = int(0)
        # filter some of the result with acreage threshold
        for c in contours:
            if cv2.contourArea(c) < self._min_box_threshold:
                continue
            else:
                (x, y, w, h) = cv2.boundingRect(c)
                all_boxes[index] = (x, y, w, h)
                index = index + 1
        boxes = [[0 for i in range(4)] for j in range(index)]
        for i in range(index):
            boxes[i] = all_boxes[i]
        return self.box_filter(boxes)

    # The method to reset acreage threshold
    def set_box_threshold(self, threshold):
        self._min_box_threshold = threshold
