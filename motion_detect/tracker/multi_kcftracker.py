from motion_detect.tracker.cnn import kcftracker as kcf
# when you want non-cnn feature, uncomment next line and comment last line
# from motion_detect.tracker import multi_kcftracker as mkcf

class KCFMultiTracker:
    def __init__(self, fixed_window=True, multiscale=True):
        self._rois = []
        self._trackers = []
        self._fixed_window = fixed_window
        self._multiscale = multiscale

    def init(self, boxes, image):
        self._trackers = []
        self._rois = [boxes[i] for i in range(len(boxes))]
        for i in range(len(boxes)):
            new_tracker = kcf.KCFTracker(self._fixed_window, self._multiscale)
            new_tracker.init(boxes[i], image)
            self._trackers.append(new_tracker)

    def update(self, image):
        success = True
        temp_success = True
        tlen = len(self._trackers)
        for i in range(len(self._trackers)):
            temp_success, self._rois[i] = self._trackers[i].update(image)
            success = success and temp_success
        return success, self._rois
