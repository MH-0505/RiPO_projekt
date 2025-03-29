import cv2 as cv


class HaarCascadesFD:
    def __init__(self, interval):
        self.face_cascade = cv.CascadeClassifier('haar_cascades/haarcascade_frontalface_alt.xml')
        self.interval = interval

    def detect_face(self, frame):
        frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        frame_gray = cv.equalizeHist(frame_gray)

        faces = self.face_cascade.detectMultiScale(frame_gray)


        return faces


