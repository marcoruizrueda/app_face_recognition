#-*- coding:utf-8 -*-
from easydict import EasyDict
import numpy as np


_C = EasyDict()

# Face detection with DSDFD_VGG
_C.NUM_CLASSES = 2
_C.THRESHOLD = 0.6 # Final confidence threshold
_C.img_mean = np.array([104., 117., 123.])[:, np.newaxis, np.newaxis].astype(
    'float32')
_C.NMS_THRESH = 0.3
_C.NMS_TOP_K = 5000
_C.TOP_K = 750
_C.CONF_THRESH = 0.05
_C.VARIANCE = [0.1, 0.2] # anchor config

# Face detection with OpenFace dlib
_C.ONE_FACE = False
