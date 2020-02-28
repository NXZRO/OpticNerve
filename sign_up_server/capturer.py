import cv2
import os
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector

PACKAGE_PATH = os.path.dirname(__file__)
TMP_PATH = PACKAGE_PATH + "/tmp/"


class Capturer:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.face_imgs = []
        self.face_embs = []
        self.capture_face_num = 0

    def capture_face(self, frame):

        face_locations = self.detector.detect(frame)

        if face_locations:

            face_embs = self.recognizer.recognize(frame, face_locations)
            if face_embs is not None:

                self.face_embs.append(face_embs[0])
                self.face_imgs.append(frame)
                self.capture_face_num += 1
                self.__draw_face(frame, face_locations)

                return True

        return False

    def detect_face(self, frame):

        face_locations = self.detector.detect(frame)

        frame = self.__draw_face(frame, face_locations)

        return frame

    def __draw_face(self, frame, face_locations):
        color = (255, 0, 0)
        for face_loc in face_locations:
            (x, y, w, h) = face_loc
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)  # draw face box

        return frame
