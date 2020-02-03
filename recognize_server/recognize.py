import cv2
from face_recognize.face_detector import FaceDetector
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_tracker import FaceTracker
from database_server.db_server import DataBaseServer


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

        self.db_server = DataBaseServer()
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.face_tracker = None

        self.db_server.build_database()
        self.db_server.load_database()

    def recognize(self, frame):
        self.frame = frame
        self.__detect()

        if self.curr_face_num:  # there are faces in the frame

            if self.tracker_flag == 0 or self.curr_face_num != self.prev_face_num:
                # no tracker or no new face in the frame
                print("recognizing ...")
                self.__recognize()
                self.process_state = RecognizingState
                self.__draw_face_info()  # draw frame

            else:
                print("tracking ...")
                self.__track()
                self.process_state = TrackingState
                self.__draw_face_info()  # draw frame

        else:
            print("detecting ...")
            self.tracker_flag = 0
            self.process_state = DetectingState
            self.__draw_face_info()  # draw frame

        return self.frame

    def __detect(self):
        self.face_locations = self.face_detector.detect(self.frame)                  # face detect
        self.curr_face_num = len(self.face_locations)
        self.face_IDs = []

    def __recognize(self):
        face_embs = self.face_recognizer.recognize(self.frame, self.face_locations)  # face recognize

        if face_embs is not None:
            print("searching database...")
            self.face_IDs = self.db_server.search_database(face_embs)                # search data base

            print("setting tracker...")
            self.tracker_flag = 1                                                    # set next frame use tracking
            self.prev_face_num = self.curr_face_num                                  # record face num

            self.face_dict = {id: loc for id, loc in zip(self.face_IDs, self.face_locations)}
            self.face_tracker = FaceTracker(self.frame, self.face_dict)              # initial trackers

    def __track(self):
        ok, self.face_dict = self.face_tracker.track(self.frame)  # face track

        self.face_IDs, self.face_locations = self.__unzip_dict(self.face_dict)

        if not ok:
            self.tracker_flag = 0  # set next frame use recognizing

    def __unzip_dict(self,dict):
        ks = []
        vs = []
        for k, v in dict.items():
            ks.append(k)
            vs.append(v)
        return ks, vs

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
            for (face_ID, face_loc) in zip(self.face_IDs, self.face_locations):
                (x, y, w, h) = face_loc
                # draw face box
                cv2.rectangle(self.frame, (x, y), (x + w, y + h), color, 2)

                # draw face id
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
