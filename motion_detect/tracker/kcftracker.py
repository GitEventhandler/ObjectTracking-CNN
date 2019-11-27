import numpy as np
import cv2
from motion_detect.util import recttools, ffttools


# KCF motion_detect
class KCFTracker:
    def __init__(self, fixed_window=True, multiscale=False):
        hog = False
        # the threshold for kcf tracing itself
        self._probability_threshold = 0.85
        self._lambda = 0.0001  # regularization
        self._padding = 2.5  # extra area surrounding the target
        self._output_sigma_factor = 0.125  # bandwidth of gaussian target
        self._is_inithann = False  # is hann window initialized
        self._template_size_arr = [0, 0]  # cv::Size, [width,height]  #[int,int]
        self._roi = [0.0, 0.0, 0.0, 0.0]  # cv::Rect2f, [x,y,width,height]  #[float,float,float,float]
        self._size_patch_info = [0, 0, 0]  # [height,width,depth] [int,int,int]
        self._scale = 1.0  # float
        self._alphaf = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._prob = None  # numpy.ndarray    (size_patch[0], size_patch[1], 2)
        self._feature_avg = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        self._hann_window = None  # numpy.ndarray    raw: (size_patch[0], size_patch[1])   hog: (size_patch[2], size_patch[0]*size_patch[1])
        # VOT,if using PAMI then interp_factor = 0.02 and sigma = 0.5
        self._interp_factor = 0.075  # linear interpolation factor for adaptation
        self._sigma = 0.2  # gaussian kernel bandwidth

        if (hog):  # HOG feature
            self._cell_size = 4  # HOG cell size
            self._hog_features = True
        else:  # raw gray-scale image # aka CSK motion_detect
            self._cell_size = 1
            self._hog_features = False

        if (multiscale):
            self._template_size_val = 96  # template size
            self._scale_step = 1.05  # scale step for multi-scale estimation
            self._scale_weight = 0.96  # to downweight detection scores of other scales for added stability
        elif (fixed_window):
            self._template_size_val = 96
            self._scale_step = 1
        else:
            self._template_size_val = 1
            self._scale_step = 1

    def get_sub_pixel_peak(self, left, center, right):
        divisor = 2 * center - right - left  # float
        return (0 if abs(divisor) < 1e-3 else 0.5 * (right - left) / divisor)

    def init_hanning_window(self):
        padded_w = self._roi[2] * self._padding
        padded_h = self._roi[3] * self._padding

        if (self._template_size_val > 1):
            if (padded_w >= padded_h):
                self._scale = padded_w / float(self._template_size_val)  # resize edge to 96px，_scale is ratio
            else:
                self._scale = padded_h / float(self._template_size_val)  # patch size

            self._template_size_arr[0] = int(padded_w / self._scale)
            self._template_size_arr[1] = int(padded_h / self._scale)
        else:
            self._template_size_arr[0] = int(padded_w)
            self._template_size_arr[1] = int(padded_h)
            self._scale = 1.

        if self._hog_features:
            self._template_size_arr[0] = int(self._template_size_arr[0]) // (
                    2 * self._cell_size) * 2 * self._cell_size + 2 * self._cell_size
            self._template_size_arr[1] = int(self._template_size_arr[1]) // (
                    2 * self._cell_size) * 2 * self._cell_size + 2 * self._cell_size
        else:
            self._template_size_arr[0] = int(self._template_size_arr[0]) // 2 * 2
            self._template_size_arr[1] = int(self._template_size_arr[1]) // 2 * 2

    def create_hanning_mats(self):
        hann2t, hann1t = np.ogrid[0:self._size_patch_info[0], 0:self._size_patch_info[1]]

        hann1t = 0.5 * (1 - np.cos(2 * np.pi * hann1t / (self._size_patch_info[1] - 1)))
        hann2t = 0.5 * (1 - np.cos(2 * np.pi * hann2t / (self._size_patch_info[0] - 1)))
        hann2d = hann2t * hann1t

        if (self._hog_features):
            hann1d = hann2d.reshape(self._size_patch_info[0] * self._size_patch_info[1])
            self._hann_window = np.zeros((self._size_patch_info[2], 1), np.float32) + hann1d
        else:
            self._hann_window = hann2d
        self._hann_window = self._hann_window.astype(np.float32)

    def create_gaussian_peak(self, sizey, sizex):
        syh, sxh = sizey / 2, sizex / 2
        output_sigma = np.sqrt(sizex * sizey) / self._padding * self._output_sigma_factor
        mult = -0.5 / (output_sigma * output_sigma)
        y, x = np.ogrid[0:sizey, 0:sizex]
        y, x = (y - syh) ** 2, (x - sxh) ** 2
        res = np.exp(mult * (y + x))
        return ffttools.fftd(res)

    def gaussian_correlation(self, cur_feat, avg_feat):
        if (self._hog_features):
            c = np.zeros((self._size_patch_info[0], self._size_patch_info[1]), np.float32)
            for i in range(self._size_patch_info[2]):
                x1aux = cur_feat[i, :].reshape((self._size_patch_info[0], self._size_patch_info[1]))
                x2aux = avg_feat[i, :].reshape((self._size_patch_info[0], self._size_patch_info[1]))
                # BEGIN FFTD
                caux = cv2.mulSpectrums(ffttools.fftd(x1aux), ffttools.fftd(x2aux), 0, conjB=True)
                caux = ffttools.real(ffttools.fftd(caux, True))
                # END FFTD
                c += caux
            c = ffttools.rearrange(c)
        else:
            c = cv2.mulSpectrums(ffttools.fftd(cur_feat), ffttools.fftd(avg_feat), 0, conjB=True)
            c = ffttools.real(ffttools.fftd(c, True))
            c = ffttools.rearrange(c)
        # check the dimension of the feature, only features that have the same dimension can be mixed together.
        if (cur_feat.ndim == 3 and avg_feat.ndim == 3):
            d = (np.sum(cur_feat[:, :, 0] * cur_feat[:, :, 0]) + np.sum(
                avg_feat[:, :, 0] * avg_feat[:, :, 0]) - 2.0 * c) / (
                        self._size_patch_info[0] * self._size_patch_info[1] * self._size_patch_info[2])
        # 2d feature
        elif (cur_feat.ndim == 2 and avg_feat.ndim == 2):
            d = (np.sum(cur_feat * cur_feat) + np.sum(avg_feat * avg_feat) - 2.0 * c) / (
                    self._size_patch_info[0] * self._size_patch_info[1] * self._size_patch_info[2])
        else:
            raise Exception('Array dim error.', cur_feat.ndim)
        # kernel calulate
        d = d * (d >= 0)
        d = np.exp(-d / (self._sigma * self._sigma))
        return d

    def get_features(self, image, scale_adjust=1.0):
        extracted_roi = [0, 0, 0, 0]
        cx = self._roi[0] + self._roi[2] / 2  # float
        cy = self._roi[1] + self._roi[3] / 2  # float
        # choose the sub-picture of the origion picture
        extracted_roi[2] = int(scale_adjust * self._scale * self._template_size_arr[0])
        extracted_roi[3] = int(scale_adjust * self._scale * self._template_size_arr[1])
        extracted_roi[0] = int(cx - extracted_roi[2] / 2)
        extracted_roi[1] = int(cy - extracted_roi[3] / 2)
        sub_window = recttools.subwindow(image, extracted_roi, cv2.BORDER_REPLICATE)
        # resize it to 96px
        if (sub_window.shape[1] != self._template_size_arr[0] or
                sub_window.shape[0] != self._template_size_arr[1]):
            sub_window = cv2.resize(sub_window, tuple(self._template_size_arr))
        if (self._hog_features):
            features_map = sub_window
            # normalize the feature to [-0.5, 0.5]
            features_map = features_map.astype(np.float32) / 255.0 - 0.5
            # the sub-picture [height,width,depth = 31]
            self._size_patch_info = [sub_window.shape[0], sub_window.shape[1], sub_window.shape[2]]
            features_map = features_map.reshape((self._size_patch_info[0] * self._size_patch_info[1],
                                                 self._size_patch_info[2])).T
        # In CN feature, convert the image to grey level image
        else:
            if (sub_window.ndim == 3 and sub_window.shape[2] == 3):
                features_map = cv2.cvtColor(sub_window, cv2.COLOR_BGR2GRAY)
            elif (sub_window.ndim == 2):
                features_map = sub_window  # (size_patch[0], size_patch[1]) #np.int8  #0~255
            # normalize the feature to [-0.5, 0.5]
            features_map = features_map.astype(np.float32) / 255.0 - 0.5
            # the sub-picture [height,width,depth = 1]
            self._size_patch_info = [sub_window.shape[0], sub_window.shape[1], 1]

        if (self._is_inithann):
            self.create_hanning_mats()

        features_map = self._hann_window * features_map
        return features_map

    def detect(self, avg_feat, cur_feat):
        k = self.gaussian_correlation(cur_feat, avg_feat)
        res = ffttools.real(ffttools.fftd(ffttools.complexMultiplication(self._alphaf, ffttools.fftd(k)), True))
        _, peak_value, _, peak_index = cv2.minMaxLoc(res)  # peak_value:float  peak_index:tuple of int
        p = [float(peak_index[0]), float(peak_index[1])]  # cv::Point2f, [x,y]  #[float,float]

        if (peak_index[0] > 0 and peak_index[0] < res.shape[1] - 1):
            p[0] += self.get_sub_pixel_peak(res[peak_index[1], peak_index[0] - 1], peak_value,
                                            res[peak_index[1], peak_index[0] + 1])
        if (peak_index[1] > 0 and peak_index[1] < res.shape[0] - 1):
            p[1] += self.get_sub_pixel_peak(res[peak_index[1] - 1, peak_index[0]], peak_value,
                                            res[peak_index[1] + 1, peak_index[0]])

        p[0] -= res.shape[1] / 2.
        p[1] -= res.shape[0] / 2.
        return p, peak_value

    def train(self, feat_avg, train_interp_factor):
        k = self.gaussian_correlation(feat_avg, feat_avg)
        alphaf = ffttools.complexDivision(self._prob, ffttools.fftd(k) + self._lambda)
        self._feature_avg = (1 - train_interp_factor) * self._feature_avg + train_interp_factor * feat_avg
        self._alphaf = (1 - train_interp_factor) * self._alphaf + train_interp_factor * alphaf

    def init(self, roi, image):
        self._roi = list(map(float, roi))
        assert (roi[2] > 0 and roi[3] > 0)
        self.init_hanning_window()
        self._is_inithann = True
        # the weighted average of the feature
        self._feature_avg = self.get_features(image)
        self._prob = self.create_gaussian_peak(self._size_patch_info[0], self._size_patch_info[1])
        self._alphaf = np.zeros((self._size_patch_info[0], self._size_patch_info[1], 2), np.float32)
        self.train(self._feature_avg, 1.0)

    def update(self, image):
        # Whether tracker is successfully tracking
        success = True
        # fix edge
        if (self._roi[0] + self._roi[2] <= 0):  self._roi[0] = -self._roi[2] + 1
        if (self._roi[1] + self._roi[3] <= 0):  self._roi[1] = -self._roi[3] + 1
        if (self._roi[0] >= image.shape[1] - 1):  self._roi[0] = image.shape[1] - 2
        if (self._roi[1] >= image.shape[0] - 1):  self._roi[1] = image.shape[0] - 2
        # get center of the roi box
        cx = self._roi[0] + self._roi[2] / 2.
        cy = self._roi[1] + self._roi[3] / 2.

        loc, peak_value = self.detect(self._feature_avg, self.get_features(image))

        if (self._scale_step != 1):
            # Test at a smaller _scale
            new_loc1, new_peak_value1 = self.detect(self._feature_avg,
                                                    self.get_features(image, 1.0 / self._scale_step))
            # Test at a bigger _scale
            new_loc2, new_peak_value2 = self.detect(self._feature_avg, self.get_features(image, self._scale_step))

            if (self._scale_weight * new_peak_value1 > peak_value and new_peak_value1 > new_peak_value2):
                loc = new_loc1
                peak_value = new_peak_value1
                self._scale /= self._scale_step
                self._roi[2] /= self._scale_step
                self._roi[3] /= self._scale_step
            elif (self._scale_weight * new_peak_value2 > peak_value):
                loc = new_loc2
                peak_value = new_peak_value2
                self._scale *= self._scale_step
                self._roi[2] *= self._scale_step
                self._roi[3] *= self._scale_step
        # loc is the shift px from center of the roi box
        self._roi[0] = cx - self._roi[2] / 2.0 + loc[0] * self._cell_size * self._scale
        self._roi[1] = cy - self._roi[3] / 2.0 + loc[1] * self._cell_size * self._scale
        # edge checking
        if (self._roi[0] >= image.shape[1] - 1):
            self._roi[0] = image.shape[1] - 1
        #
        if (self._roi[1] >= image.shape[0] - 1):
            self._roi[1] = image.shape[0] - 1
        #
        if (self._roi[0] + self._roi[2] <= 0):
            self._roi[0] = -self._roi[2] + 2
        #
        if (self._roi[1] + self._roi[3] <= 0):
            self._roi[1] = -self._roi[3] + 2
        assert (self._roi[2] > 0 and self._roi[3] > 0)

        new_feat_avg = self.get_features(image)
        self.train(new_feat_avg, self._interp_factor)

        if peak_value < self._probability_threshold:
            success = False
        return success, self._roi
