import cv2
import gui.util.cv_rectutil as cvrect
import gui.extra_function.area_monitor as amonitor
import gui.extra_function.structs as structs
from motion_detect import motion_detector as motdet

white = (255, 255, 255)
red = (255, 0, 0)
green = (0, 255, 0)
blue = (0, 0, 255)
black = (0, 0, 0)


class BackendGuiBridge:
    def __init__(self, video_path, lines=[], areas=[], draw_trace=True):
        self._video_cap = cv2.VideoCapture(video_path)
        self._area_monitor = amonitor.AreaMonitor()
        self._draw_trace = draw_trace
        self._current_frame_index = 0
        self._motion_detector = motdet.MotionDetector()
        self._log_list = []
        self._rect_touched_area = 0
        self._rect_touched_line = 0
        self._all_frame_rect_centers = []
        for l in lines:
            sline = structs.Line()
            sline.from_cv_line(l)
            self._area_monitor.add_line(sline)
        for a in areas:
            sarea = structs.Rectangle()
            sarea.from_cv_rect(a)
            self._area_monitor.add_rect(sarea)

    def __del__(self):
        self._video_cap.release()

    # add tripwire
    def add_tripwire(self, line: structs.Line):
        self._area_monitor.add_line(line)

    # add forbidden area
    def add_area(self, area: structs.Rectangle):
        sarea = structs.Rectangle()
        sarea.from_cv_rect(area)
        self._area_monitor.add_rect(sarea)

    # print log touched log
    def _print_line_log(self, rect: structs.Rectangle, line: structs.Line):
        r = rect.to_cv_rect()
        p0 = list(line.a_point.to_cv_point())
        p1 = list(line.b_point.to_cv_point())
        self._log_list.append(
            'target ({0},{1},{2},{3}) has touched the line [({4},{5}),({6},{7})]'.format(int(r[0]), int(r[1]),
                                                                                         int(r[2]), int(r[3]),
                                                                                         int(p0[0]), int(p0[1]),
                                                                                         int(p1[0]),
                                                                                         int(p1[1])))

    # Print area entrance log
    def _print_area_log(self, rect: structs.Rectangle, area: structs.Rectangle):
        r = rect.to_cv_rect()
        a = area.to_cv_rect()
        self._log_list.append(
            'target ({0},{1},{2},{3}) has entered the area({4},{5},{6},{7})'.format(int(r[0]), int(r[1]), int(r[2]),
                                                                                    int(r[3]),
                                                                                    int(a[0]), int(a[1]), int(a[2]),
                                                                                    int(a[3])))

    def _draw_area(self, frame):
        for line in self._area_monitor.get_lines():
            cvrect.draw_line(frame, line.a_point.to_cv_point(), line.b_point.to_cv_point(), blue)
        for area in self._area_monitor.get_rects():
            cvrect.draw_rect(frame, area.to_cv_rect(), blue)
        return frame

    def _draw_tracking_rects(self, frame, rect_problems=[]):
        # clear log
        self._log_list.clear()
        # check warnings
        for rect_problem in rect_problems:
            who = rect_problem.which_rect
            # tripwire warnings
            for line in rect_problem.line_touched:
                self._print_line_log(who, line)
            # area warnings
            for area in rect_problem.rect_trespassed:
                self._print_area_log(who, area)
            # change different color for different target
            if len(rect_problem.rect_trespassed) > 0 or len(rect_problem.line_touched) > 0:
                cvrect.draw_rect(frame, who.to_cv_rect(), red)
            else:
                cvrect.draw_rect(frame, who.to_cv_rect(), green)
        return frame

    # get a pre-processed frame
    def get_frame(self):
        success, frame = self._video_cap.read()
        if success:
            self._current_frame_index = self._current_frame_index + 1
            rects = self._motion_detector.detect(frame)
            if self._draw_trace:
                self._all_frame_rect_centers.append(self._get_rect_centers(rects))
            rect_problems = self._area_monitor.check_problems(rects)
            frame = self._draw_area(frame)
            frame = self._draw_tracking_rects(frame, rect_problems)
            if self._draw_trace:
                frame = self._draw_trace(frame)
            return success, frame
        else:
            return success, None

    def get_logs(self):
        return self._log_list

    def _get_rect_centers(self, rects=[]):
        current_frame_center = []
        for x, y, a, b in rects:
            center = [x + a / 2, y + b / 2]
            current_frame_center.append(center)
        return current_frame_center
