import cv2
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector
from database_server.db_server import DataBaseServer


DetectingState = 0
RecognizingState = 1
TrackingState = 2


class RecognizeServer:
    def __init__(self):
        self.frame = None
        self.face_locations = []
        self.curr_face_num = 0
        self.face_IDs = []
        self.process_state = None

        self.db_server = DataBaseServer()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()

        self.db_server.build_database()
        self.db_server.load_database()

    def recognize(self, frame):
        self.frame = frame
        self.face_locations = self.face_detector.detect(frame)  # face detect

        self.curr_face_num = len(self.face_locations)

        self.face_IDs = ["Detect" + str(i) for i in range(self.curr_face_num)]

        if self.curr_face_num:  # there are faces in the frame
            print("recognizing...")

            face_embs = self.face_recognizer.recognize(self.frame, self.face_locations)  # face recognize
            if face_embs is not None:
                print("searching database...")
                self.face_IDs = self.db_server.search_database(face_embs)  # search data base
                self.process_state = RecognizingState

            self.draw_face_info()  # draw frame

        else:
            print("detecting ...")
            self.process_state = DetectingState
            self.draw_face_info()  # draw frame

        return self.frame

    def draw_face_info(self):
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
            for (face_ID, face_loc) in zip(self.face_IDs, self.face_locations):
                (x, y, w, h) = face_loc
                # draw face box
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)

                # draw face id
                cv2.putText(self.frame, str(face_ID), (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                            1, color, 1, cv2.LINE_AA)

        else:
            for face_loc in self.face_locations:
                (x, y, w, h) = face_loc
                # draw face box
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)


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
    FaceDetector()
