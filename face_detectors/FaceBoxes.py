# Face detection by FaceBoxes-Tensorflow2

#import time
#import os
#os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'

import numpy as np
from PIL import Image, ImageDraw
import cv2
import numpy as np
from .FaceBoxes_tensorflow.face_detector import FaceDetector
import dlib

MODEL_PATH = 'face_detectors/FaceBoxes_tensorflow/model.pb'

def find_max_face(bbs):
    if bbs == None or len(bbs) == 1: 
        return 0
    else:
        areas = []
        for b in bbs:
            a = b.width() * b.height()
            areas.append(a)
        if len(areas) > 0:
            return np.argmax(areas)    
    return 0 

def draw_boxes_on_image(image, boxes, scores):
    bbs = []
    for b, s in zip(boxes, scores):
        ymin, xmin, ymax, xmax = b
        cv2.rectangle(image, (xmin, ymin), (xmax, ymax), [255,255,0], 2)
        cv2.putText(image, '{:.3f}'.format(s), (xmin, ymin), fontFace=cv2.FONT_HERSHEY_DUPLEX, 
                    fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
        bb = dlib.rectangle(xmin, ymin, xmax, ymax)
        bbs.append(bb)
    return image, bbs

def main(rgbImage):
    #start = time.time()
    face_detector = FaceDetector(MODEL_PATH, gpu_memory_fraction=0.25, visible_device_list='0')

    #latency = (time.time() - start)*1000
    #print("Detection took {} milliseconds".format(latency))
    boxes, scores = face_detector(rgbImage, score_threshold=0.3)
    rgbImage, bbs = draw_boxes_on_image(rgbImage, boxes, scores)
    
    if len(bbs) <= 0 or bbs[0] is None:
        print('No faces detected')
        return rgbImage, None, 0
    return rgbImage, bbs, find_max_face(bbs)

