import cv2
import os
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector

PACKAGE_PATH = os.path.dirname(__file__)
TMP_PATH = PACKAGE_PATH + "/tmp/"


class FaceCapturer:
    def __init__(self):
        self.detector = FaceDetector()
        self.recognizer = FaceRecognizer()
        self.frame = None
        self.face_imgs = []
        self.face_locations = []
        self.face_embs = []
        self.capture_face_num = 0

    def capture_face(self):
        self.face_imgs = []
        self.face_locations = []
        self.face_embs = []
        self.capture_face_num = 0

        camera = cv2.VideoCapture(0)  # 0 -> first camera

        while self.capture_face_num < 3:

            ret, self.frame = camera.read()  # get frame

            face_emb = self.__embedding_face()

            cv2.imshow('frame', self.frame)  # show frame in window

            # press 'q' to stop
            if cv2.waitKey(1) & 0xFF == ord('q') and face_emb is not None:
                self.face_embs.append(face_emb)
                self.face_imgs.append(self.frame)
                cv2.imwrite(TMP_PATH + str(self.capture_face_num) + ".jpg", self.frame)
                self.capture_face_num += 1

        camera.release()  # camera release
        cv2.destroyAllWindows()  # close windows

        return self.face_embs

    def capture_test_imgs(self, dir_name):
        self.face_imgs = []
        self.face_locations = []
        self.face_embs = []
        self.capture_face_num = 0
        for img_file in os.listdir(dir_name):
            self.frame = cv2.imread(dir_name + img_file)
            face_emb = self.__embedding_face()
            if face_emb is not None:
                self.face_embs.append(face_emb)
                self.face_imgs.append(self.frame)

        return self.face_embs

    def __embedding_face(self):
        self.face_locations = self.detector.detect(self.frame)

        if self.face_locations:
            face_embs = self.recognizer.recognize(self.frame, self.face_locations)
            face_emb = face_embs[0]
            self.__draw_face()
        else:
            face_emb = None

        return face_emb

    def __draw_face(self):
        color = (255, 0, 0)
        for face_loc in self.face_locations:
            (x, y, w, h) = face_loc
            # draw face box
            cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)

        # draw current face num
        cv2.putText(self.frame, "cap_face_num : " + str(self.capture_face_num), (0, 60), cv2.FONT_HERSHEY_DUPLEX,
                    1, color, 1, cv2.LINE_AA)


if __name__ == '__main__':
    face_cap = FaceCapturer()
    face_embs = face_cap.capture_face()
    print(face_embs)
