#-*- coding:utf-8 -*-
from easydict import EasyDict
import numpy as np


_C = EasyDict()

# General
_C.FACE_TOLERANCE = 0.6 # Greater than or equal to this value are considered positive recognition

# Face recognition with OpenFace dlib
_C.PATH_DATASET = '/home/marco/face_recognizer/openface/training-images/lfw/raw/1-raw/'

