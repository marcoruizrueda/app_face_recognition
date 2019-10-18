import sys
#sys.path.append('/home/marco/face_recognizer/marco_env/lib/python3.5/site-packages/')
sys.path.append('/usr/lib/python3/dist-packages')
import cv2
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QThread, pyqtSignal, Qt
from PyQt4.QtGui import QImage, QPixmap #QLabel, 
import time
import threading
from users import utils


# Face detectors
from face_detectors import DSFD_VGG2, cvlibCNN, FaceBoxes, OpenFace_dlib

# Facial recognizers
from facial_recognizers import OpenFace
 
qtCreatorFile = "main_GUI.ui" # Enter file of PyQT4 UI here.
cameraSource = 0
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class videoThread(QThread):
    
    changePixmap = pyqtSignal(QImage)
    changeLabel = pyqtSignal(list)
    def __init__(self, face_method=None, recognition_method=None):
        super(videoThread,self).__init__()
        self.face_method = face_method
        self.recognition_method = recognition_method
    
    def run(self):
        
        def initialize_labels():
            self.persons =  ['--']
            self.person = '--'
            self.confidences = ['--']
            self.confidence = '--'
            self.gender = '--'
            self.position = '--'
            self.project = '--'
            self.advisor = '--'
        
        cap = cv2.VideoCapture(cameraSource)
        cap.set(cv2.CAP_PROP_FPS, 20)
        #print('FPS: ', cap.get(cv2.CAP_PROP_FPS))
        #cap.set(3, 320)
        #cap.set(4, 240)
        self.app = MyApp()
        starttime = time.time()
        refresh_rate = 3 # Seconds to repaint GUI labels
        acumtemp = 0
        while cap.isOpened():
            ret,frame = cap.read()
            if ret != True: break
            
            start = time.time()

            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            orig_rgbImage = rgbImage.copy()
            bbs = None
            '''persons =  ['Name']
            confidences = ['--']
            gender = '--'
            position = '--'
            project = '--'
            advisor = '--' 
            '''
            index_max = 0
            initialize_labels()
            
            # Run face detection
            if self.face_method == 'DSFD_VGG':
                rgbImage, bbs, index_max = DSFD_VGG2.main(rgbImage)
            elif self.face_method == 'Cvlib':
                rgbImage, bbs, index_max = cvlibCNN.main(rgbImage)
            elif self.face_method == 'FaceBoxes':
                rgbImage, bbs, index_max = FaceBoxes.main(rgbImage)
            elif self.face_method == 'OpenFace dlib':
                OpenFace_dlib.is_one_face = False
                rgbImage, bbs, index_max = OpenFace_dlib.main(rgbImage)
            elif self.face_method == 'OpenFace dlib (one face)':
                OpenFace_dlib.is_one_face = True
                rgbImage, bbs, index_max = OpenFace_dlib.main(rgbImage)
            
            # Run facial recognition
            if (bbs is None or len(bbs)==0) and \
                self.recognition_method != 'Select method...' and \
                self.recognition_method != None:
                print('No faces to recognize. Select a Face detector...')
            elif self.recognition_method == 'OpenFace (SVM)':
                rgbImage, self.persons, self.confidences = OpenFace.main(orig_rgbImage, bbs)
                
            
            latency = time.time() - start
            total_FPS = float('{:.2f}'.format(1 / latency))
            total_FPS = min(total_FPS, cap.get(cv2.CAP_PROP_FPS))
            #print('Total FPS: {:.2f}'.format(total_FPS))

            # Show result in labels
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            #p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(convertToQtFormat)
            
            # Update labels each refresh_rate seconds
            finaltime = time.time() - starttime
            if finaltime > acumtemp: 
                #print("Jump on: ", finaltime)
                if bbs is None or len(bbs)==0 :
                    initialize_labels()
                else:
                    if len(self.persons) >= index_max+1: 
                        self.person = self.persons[index_max]
                        self.confidence = self.confidences[index_max]
                        if str(self.confidence).replace('.','',1).isdigit(): 
                            self.confidence = float('{:.2f}'.format(float(self.confidences[index_max])))
                    
                    # Import person's information from TXT file
                    self.gender, self.position, self.project, self.advisor = utils.read_txtUsers(self.person)
                    
                self.changeLabel.emit([str(total_FPS), self.person , self.confidence, self.gender, self.position, self.project, self.advisor])
                
                acumtemp = acumtemp + refresh_rate
                
 
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.processFrame() # Start streaming
        self.cbFaceDetector.currentIndexChanged.connect(self.on_cbFace_changed)
        self.cbFaceRecognizer.currentIndexChanged.connect(self.on_cbFacial_changed)
        self.on_label_img_change()
        #self.pushButton.clicked.connect(self.processFrameD) 
    
    def on_cbFace_changed(self, value):
        self.video.face_method = self.cbFaceDetector.currentText()
    
    def on_cbFacial_changed(self, value):
        self.video.recognition_method = self.cbFaceRecognizer.currentText()
    
    def processFrame(self):        
        self.video = videoThread(None)
        self.video.start()
        self.video.changePixmap.connect(self.setFrame) # Show resultant image
    
    def setFrame(self,frame):
        pixmap = QPixmap.fromImage(frame)
        self.label_img.setPixmap(pixmap)
    
    def setLabel(self,labels):
        #print("Entra ", label)
        self.label_fps.setText(labels[0]) # FPS
        self.label_fullname.setText(str(labels[1])) # Name
        self.label_confidence.setText(str(labels[2])) # Confidence
        self.label_gender.setText(str(labels[3])) # Gender
        self.label_position.setText(str(labels[4])) # Position
        self.label_project.setText(str(labels[5])) # Project
        self.label_advisor.setText(str(labels[6])) # Advisor
        
    def on_label_img_change(self):
        self.video.changeLabel.connect(self.setLabel) # Show resultant label
        
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
