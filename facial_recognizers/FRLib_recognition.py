# Face detection by cvlib
import cv2
import numpy as np
import dlib
import face_recognition
import os
import glob
from facial_recognizers import config as cfg
'''import sys
sys.path.append('../users')
from users import utils'''
#import time

def get_noRepeatedFiles():
    files = [f for f in glob.glob(cfg._C.PATH_DATASET + "**/*.jpg", recursive=False)]

    known_face_encodings = []
    known_face_names = []
    for f in files:
        dirname = os.path.basename(os.path.dirname(f))
        if dirname in known_face_names:
            continue
        
        # Check if face is reliable for encoding
        person_image = face_recognition.load_image_file(f)
        person_face_encoding = face_recognition.face_encodings(person_image)
        if len(person_face_encoding)>0: 
            person_face_encoding = person_face_encoding[0]
            known_face_encodings.append(person_face_encoding)
            known_face_names.append(dirname)
            #print(dirname, f)
        else:
            continue
        
    return known_face_encodings, known_face_names

# Create arrays of known face encodings and their names
known_face_encodings, known_face_names = get_noRepeatedFiles()


def predict_face(face_encoding):
    # See if the face is a match for the known face(s)
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding, cfg._C.FACE_TOLERANCE)
    name = "Unknown"

    # Use the known face with the smallest distance to the new face
    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    best_match_index = np.argmin(face_distances)
    if matches[best_match_index]:
        name = known_face_names[best_match_index].replace('_', ' ')

    distance = face_distances[best_match_index]
    confidence = 1/(1+distance)
    return name, confidence

def main(rgbImage, bbs):
    # Resize frame of video to 1/4 size for faster face recognition processing
    small_frame = cv2.resize(rgbImage, (0, 0), fx=0.25, fy=0.25)
    
    # Transform dlib bbs to FRLib bbs 
    face_locations = [(int(face.top()/4), int(face.right()/4), int(face.bottom()/4), int(face.left()/4)) for face in bbs]
    #cv2.rectangle(small_frame, (face_locations[0][0], face_locations[0][1]), (face_locations[0][2], face_locations[0][3]), (0, 0, 255), cv2.FILLED)
    #cv2.imwrite('/home/marco/Downloads/rr.jpg', small_frame)
    # Find all the faces and face encodings in the current frame of video
    face_encodings = face_recognition.face_encodings(small_frame, face_locations)

    persons = []
    confidences = []
    for face_encoding in face_encodings:
        name, confidence = predict_face(face_encoding)
        persons.append(name)
        confidences.append(confidence)
    
    # Display the results
    for i, ((top, right, bottom, left), name) in enumerate(zip(face_locations, persons)):
        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
        bottom += 20
        
        # Draw a box around the face
        cv2.rectangle(rgbImage, (left, top), (right, bottom), (0, 255, 0), 2)

        # Draw a label with a name below the face
        #cv2.rectangle(rgbImage, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        #font = cv2.FONT_HERSHEY_DUPLEX
        #cv2.putText(rgbImage, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        text = name + '(' + str(round(confidences[i],2)) + ')'
        cv2.putText(rgbImage, text, (left, top-5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)
    
    return rgbImage, persons, confidences