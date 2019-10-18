# Face detection by dlib from OpenFace
#-*- coding:utf-8 -*-
from face_detectors import config as cfg
#import time
import cv2
import sys
sys.path.append('/usr/local/lib/python3.5/dist-packages/')
import openface
import os
import numpy as np

modelsDir  = '/home/marco/face_recognizer/openface/models/'
align = openface.AlignDlib(os.path.join(modelsDir, 'dlib', 
                                        "shape_predictor_68_face_landmarks.dat"))
is_one_face = cfg._C.ONE_FACE

def find_max_face(bbs):
	if bbs == None or len(bbs) == 1: 
		return 0
	else:
		areas = []
		for b in bbs:
			areas.append(b.area())
		return np.argmax(areas)    
	return None


def draw_prediction(img, bbs):
	for face in bbs:
		x = face.left()
		y = face.top()
		w = face.right() - x
		h = face.bottom() - y
		cv2.rectangle(img, (x,y), (x+w,y+h), (255,255,0), 2)
	return img


def main(rgbImg):
	rgbImg2 = rgbImg.copy()
	#rgbImg = cv2.cvtColor(rgbImg, cv2.COLOR_BGR2RGB)
	#start = time.time()
	bbs = []
	if is_one_face:
		bbs = [align.getLargestFaceBoundingBox(rgbImg2)]
	else:
		bbs = align.getAllFaceBoundingBoxes(rgbImg2)
	#latency = (time.time() - start)*1000
	#print("Detection took {} milliseconds".format(latency))
	
	if len(bbs) <= 0 or bbs[0] is None:
		print('No faces detected')
		return rgbImg, None, 0
	else:
		rgbImg = draw_prediction(rgbImg2, bbs)

	return rgbImg, bbs, find_max_face(bbs)
