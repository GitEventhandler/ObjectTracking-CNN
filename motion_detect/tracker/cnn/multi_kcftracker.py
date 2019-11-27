from motion_detect.tracker.cnn import kcftracker as kcf
import numpy as np
import cv2
import torch
from torchvision.models.vgg import vgg16

# Conv1_2 data from vgg16 net
VGG16 = vgg16(pretrained=True).features[:2]
if torch.cuda.is_available():
    VGG16 = VGG16.cuda()
    print('Using GPU Acceleration')
else:
    print('Using Only CPU')


def get_features(z):
    VGG16.eval()
    image = cv2.resize(z, (256, 256))
    if np.max(image) <= 1:
        # image = (image * 255).astype(theano.config.floatX)
        image = (image * 255)
    image = torch.Tensor(image)
    if torch.cuda.is_available():
        image = image.cuda()
    # print(image.shape)
    image = image.permute(2, 0, 1)
    # image = image[None, :, :, :]
    image = image.unsqueeze(0)
    cnn_feat = VGG16(image)[0]
    # print(cnn_feat.shape)
    cnn_feat = cnn_feat.sum(dim=0)
    # cnn_feat = cnn_feat[None, :, :]
    cnn_feat = cnn_feat.unsqueeze(2).detach().cpu().numpy()
    # cnn_feat = cnn_feat.transpose(1, 2, 0)
    cnn_feat = cv2.resize(cnn_feat, (z.shape[1], z.shape[0]))
    return cnn_feat


# Muti-target Tracker
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
            new_tracker = kcf.KCFTracker(self._fixed_window, self._multiscale, 'cnn')
            new_tracker.init(boxes[i], image, get_features)
            self._trackers.append(new_tracker)

    def update(self, image):
        success = True
        temp_success = True
        for i in range(len(self._trackers)):
            temp_success, self._rois[i] = self._trackers[i].update(image, get_features)
            success = success and temp_success
        return success, self._rois
