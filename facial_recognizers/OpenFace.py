# Facial recognition by openface
import numpy as np
import cv2
import os
import sys
sys.path.append('/usr/local/lib/python3.5/dist-packages/')
import openface
#import time
import pickle

modelsDir  = '/home/marco/face_recognizer/openface/models/'
networkModel = os.path.join(modelsDir, 'openface', 'nn4.small2.v1.t7')
imgDim = 96
net = openface.TorchNeuralNet(networkModel, imgDim=imgDim, cuda=False)
classifierModel = '/home/marco/face_recognizer/openface/training-images/lfw/feature/classifier_marco_linear.pkl'
align = openface.AlignDlib(os.path.join(modelsDir, 'dlib', "shape_predictor_68_face_landmarks.dat"))


def getRep(rgbImg, bbs):
    reps = []
    for bb in bbs:
        #start = time.time()
        alignedFace = align.align(
            imgDim,
            rgbImg,
            bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image")
        #else:
        #    print("Alignment took {} seconds.".format(time.time() - start))

        #start = time.time()
        rep = net.forward(alignedFace) # Neural Network classifier
        #print("Neural network forward pass took {} seconds.".format(time.time() - start))
        reps.append((bb, rep))
    return reps

def draw_prediction(img, bbs, persons, confidences):
    #img2 = img.copy()
    for person, face in enumerate(bbs):
        x = face.left()
        y = face.top()
        w = face.right() - x
        h = face.bottom() - y
        # draw box over face
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        #text = str(persons[person][0].decode("utf-8"))
        text = persons[person]
        text = text + '(' + str(round(confidences[person],2)) + ')'
        cv2.putText(img, text, (x, y-5), fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=0.5, color=(0, 0, 255), thickness=1, lineType=cv2.LINE_AA)

    return img

def infer(frame, bbs):
    def predict_image(rgbImage, bbs):
        reps = getRep(rgbImage, bbs)
        bbs = []
        persons = []
        confidences = []
        for r in reps:
            rep = r[1].reshape(1, -1)
            bbx = r[0]
            bbs.append(bbx)
            #start = time.time()
            predictions = clf.predict_proba(rep).ravel()
            maxI = np.argmax(predictions)
            person = le.inverse_transform([maxI])
            persons.append(person[0].decode("utf-8").replace('_', ' '))
            confidence = predictions[maxI]
            confidences.append(confidence)
            #print("Prediction took {} seconds.".format(time.time() - start))
            #print("Predict {} with {:.2f} confidence.".format(person[0].decode('utf-8'), confidence))
        rgbImage = draw_prediction(rgbImage, bbs, persons, confidences)
        return rgbImage, persons, confidences

    # Predict face
    with open(classifierModel, 'rb') as f:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        (le, clf) = u.load()

    frame, persons, confidences = predict_image(frame, bbs)
    return frame, persons, confidences


def main(rgbImage, bbs):
    rgbImage, persons, confidences = infer(rgbImage, bbs)
    return rgbImage, persons, confidences