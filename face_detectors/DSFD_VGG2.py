#-*- coding:utf-8 -*-
#import sys
#sys.path.append('DSFD_pytorch/')
#sys.path.append('face_recognizer/app_face_recognition/face_detectors/DSFD_pytorch')
from face_detectors.DSFD_pytorch import demo
import cv2
import numpy as np
#import time

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

def main(rgbImage):
	#start = time.time()
    rgbImage, bbs = demo.main(rgbImage)
	#latency = (time.time() - start)*1000
	#print("Detection took {} milliseconds".format(latency))

    if len(bbs) <= 0 or bbs[0] is None:
        print('No faces detected')
        return rgbImage, None, 0
    
    return rgbImage, bbs, find_max_face(bbs)
