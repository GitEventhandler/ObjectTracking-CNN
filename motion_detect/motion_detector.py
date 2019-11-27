from motion_detect.tracker import multi_kcftracker as mkcf
from motion_detect.bma.mog2 import Mog2Detect as bma_algorithm


class MotionDetector:
    def __init__(self, reinit_interval=90):
        self._bgs_algorithm = bma_algorithm(100)
        self._tracker = mkcf.KCFMultiTracker(True, True)
        self._frame = None
        self._is_need_reinit = True
        self._current_frame_index = int(0)
        # reinit every 90 fps
        self._reinit_interval = reinit_interval
        self._rois = []

    def _reinitialize(self):
        boxes_temp = self._bgs_algorithm.get_object_boxes(self._frame)
        # Complete init need at least have one target
        if len(boxes_temp) > 0:
            self._rois = boxes_temp
            self._is_need_reinit = False
            self._tracker.init(self._rois, self._frame)

    def _update(self):
        success, boxes_temp = self._tracker.update(self._frame)
        self._rois = boxes_temp
        if not success:
            self._is_need_reinit = True

    def detect(self, frame):
        self._frame = frame
        self._current_frame_index = self._current_frame_index + 1
        if self._is_need_reinit:
            self._reinitialize()
        else:
            self._bgs_algorithm.train_only(frame)
            self._update()
        # need to reinit when meet the interval to prevent that the target been ignored
        if self._current_frame_index % self._reinit_interval == 0:
            self._is_need_reinit = True
        return self._rois
