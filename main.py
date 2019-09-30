import sys
import cv2
from PyQt4 import QtCore, QtGui, uic
from PyQt4.QtCore import QThread, pyqtSignal, Qt
from PyQt4.QtGui import QImage, QPixmap
 
qtCreatorFile = "main_GUI.ui" # Enter file of PyQT4 UI here.
cameraSource = 0
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(qtCreatorFile)
 
class videoThread(QThread):
    
    changePixmap = pyqtSignal(QImage)
        
    def __init__(self):
        super(videoThread,self).__init__()

    def run(self):        
        cap = cv2.VideoCapture(cameraSource)
        while cap.isOpened():
            _,frame = cap.read()
            rgbImage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Run recognizer
            
            
            # Show result in label
            h, w, ch = rgbImage.shape
            bytesPerLine = ch * w
            convertToQtFormat = QImage(rgbImage.data, w, h, bytesPerLine, QtGui.QImage.Format_RGB888)
            p = convertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)
            self.changePixmap.emit(p)
 
class MyApp(QtGui.QMainWindow, Ui_MainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.pushButton.clicked.connect(self.processFrame) # Start streaming
    
    def processFrame(self):        
        self.video = videoThread()
        self.video.start()
        self.video.changePixmap.connect(self.setFrame) # Show resultant image
    
    def setFrame(self,frame):
        pixmap = QPixmap.fromImage(frame)
        self.label_img.setPixmap(pixmap)
        
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MyApp()
    window.show()
    sys.exit(app.exec_())
