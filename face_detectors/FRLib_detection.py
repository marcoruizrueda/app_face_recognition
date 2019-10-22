# Face detection by cvlib
import cv2
import numpy as np
import dlib
import face_recognition
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
    
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(rgbImage, (0, 0), fx=0.25, fy=0.25)
    
    # Only process every other frame of video to save time
    face_locations = face_recognition.face_locations(small_frame)
        
    bbs = []
    for (top, right, bottom, left) in face_locations:
        # Scale back up face locations since the frame we detected in was scaled to 1/4 size
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        #bottom += 20
        cv2.rectangle(rgbImage, (left, top), (right, bottom+20), (255,255,0), 2)
        bb = dlib.rectangle(left, top, right, bottom)
        bbs.append(bb)
        
    if len(bbs) <= 0 or bbs[0] is None:
        print('No faces detected')
        return rgbImage, None, 0
    
    #latency = (time.time() - start)*1000
    #print("Detection took {} milliseconds".format(latency))
    
    return rgbImage, bbs, find_max_face(bbs)