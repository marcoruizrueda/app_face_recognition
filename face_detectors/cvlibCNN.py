# Face detection by cvlib
import cv2
import cvlib as cv
import numpy as np
import dlib
from cvlib.object_detection import draw_bbox
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
    faces, confidences = cv.detect_face(rgbImage)
    
    bbs = []
    for face,conf in zip(faces,confidences):
        (startX,startY) = face[0],face[1]
        (endX,endY) = face[2],face[3]
        cv2.rectangle(rgbImage, (startX,startY), (endX,endY), (255,255,0), 2)
        bb = dlib.rectangle(startX, startY, endX, endY)
        bbs.append(bb)
    #latency = (time.time() - start)*1000
    #print("Detection took {} milliseconds".format(latency))
    if len(bbs) <= 0 or bbs[0] is None:
        print('No faces detected')
        return rgbImage, None, 0
    
    return rgbImage, bbs, find_max_face(bbs)