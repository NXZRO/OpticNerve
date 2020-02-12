import cv2
from face_recognize.face_detector import FaceDetector
from face_recognize.face_recognizer import FaceRecognizer
from database_server.flann_server import FlannServer


DetectingState = 0
RecognizingState = 1
TrackingState = 2


class RecognizeServer:
    def __init__(self):
        self.curr_face_num = 0
        self.prev_face_num = 0
        self.tracker_flag = 0
        self.process_state = 0

        self.frame = None
        self.face_locations = []
        self.face_IDs = []
        self.face_dict = {}

        self.flann_server = FlannServer()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

        self.flann_server.build()
        self.flann_server.load()

    def recognize(self, frame):
        self.frame = frame
        self.__detect()

        if self.curr_face_num:  # there are faces in the frame
            print("recognizing ...")
            self.face_IDs = []
            self.__recognize()
            self.process_state = RecognizingState
            self.__draw_face_info()  # draw frame

        else:
            print("detecting ...")
            self.face_IDs = []
            self.tracker_flag = 0
            self.process_state = DetectingState
            self.__draw_face_info()  # draw frame

        return self.frame

    def __detect(self):
        self.face_locations = self.face_detector.detect(self.frame)                  # face detect
        self.curr_face_num = len(self.face_locations)

    def __recognize(self):
        face_embs = self.face_recognizer.recognize(self.frame, self.face_locations)  # face recognize

        if face_embs is not None:
            print("searching database_server...")
            self.face_IDs = self.flann_server.search(face_embs)                # search data base

    def __draw_face_info(self):
        color_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
        color = color_dict[self.process_state]

        state_dict = {0: "Detecting...", 1: "Recognizing...", 2: "Tracking"}
        state = state_dict[self.process_state]

        # draw process state
        cv2.putText(self.frame, state, (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                    1, color, 1, cv2.LINE_AA)

        # draw current face num
        cv2.putText(self.frame, "face_num : " + str(self.curr_face_num), (0, 60), cv2.FONT_HERSHEY_DUPLEX,
                    1, color, 1, cv2.LINE_AA)

        if self.face_IDs:
            for i, (face_ID, face_loc) in enumerate(zip(self.face_IDs, self.face_locations)):
                (x, y, w, h) = face_loc

                if face_ID == "":
                    face_ID = "Unknow" + str(i)

                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(self.frame, str(face_ID), (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                            1, color, 1, cv2.LINE_AA)


if __name__ == '__main__':

    server = RecognizeServer()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        frame = server.recognize(frame)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
