import cv2
from recognize_server.face_detector import FaceDetector
from recognize_server.face_embedder import FaceEmbedder
from database_server.flann_server import FlannServer
import time

DetectingState = 0
RecognizingState = 1


class FaceRecognizer:
    def __init__(self, recognize_flag=True):
        self.flann_server = FlannServer()
        self.face_detector = FaceDetector()
        self.face_embedder = FaceEmbedder()

        if recognize_flag is True:
            self.flann_server.build()
            self.flann_server.load()

    def recognize(self, frame):
        face_ids = []

        face_locations = self.face_detector.detect(frame)

        if face_locations:
            print("recognizing ...")

            face_embs = self.face_embedder.embedding(frame, face_locations)

            face_ids = self.flann_server.search(face_embs)

            self.__draw_face_info(frame, face_locations, face_ids, RecognizingState)

        else:
            print("detecting ...")

            self.__draw_face_info(frame, face_locations, face_ids, DetectingState)

        return frame, face_ids

    def embedding(self, frame):

        face_embs = []

        face_locations = self.face_detector.detect(frame)

        if face_locations:
            face_embs = self.face_embedder.embedding(frame, face_locations)

        self.__draw_face_boxes(frame, face_locations)

        return frame, face_embs

    @staticmethod
    def __draw_face_info(frame, face_locs, face_ids, process_state):
        color_dict = {0: (255, 0, 0), 1: (0, 255, 0)}
        color = color_dict[process_state]

        state_dict = {0: "Detecting...", 1: "Recognizing..."}
        state = state_dict[process_state]

        curr_face_num = len(face_locs)

        # draw process state
        cv2.putText(frame, state, (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                    1, color, 1, cv2.LINE_AA)

        # draw current face num
        cv2.putText(frame, "face_num : " + str(curr_face_num), (0, 60), cv2.FONT_HERSHEY_DUPLEX,
                    1, color, 1, cv2.LINE_AA)

        # draw face id and face boxes
        if face_ids:
            for i, (face_id, face_loc) in enumerate(zip(face_ids, face_locs)):
                (x1, y1, x2, y2) = face_loc

                if face_id == "":
                    face_id = "Unknow" + str(i)

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, face_id, (x1, y1 - 20), cv2.FONT_HERSHEY_DUPLEX,
                            1, color, 1, cv2.LINE_AA)

    @staticmethod
    def __draw_face_boxes(frame, face_locations):
        color = (255, 0, 0)
        for face_loc in face_locations:
            (x1, y1, x2, y2) = face_loc
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)  # draw face box

        return frame


if __name__ == '__main__':

    recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(0)  # Fish -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        frame, face_ids = recognizer.recognize(frame)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
