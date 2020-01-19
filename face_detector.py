import cv2
from mtcnn import MTCNN


class FaceDetector:
    def __init__(self):
        self.mtcnn_detector = MTCNN()
        self.inp_frame = None
        self.face_locations = []

    def detect(self, inp_frame):
        self.__read_frame(inp_frame)
        faces = self.mtcnn_detector.detect_faces(self.inp_frame)

        for face in faces:
            self.face_locations.append(face['box'])

        return self.face_locations

    def __read_frame(self, inp_frame):
        # clear params when read new frame
        self.inp_frame = inp_frame
        self.face_locations = []


if __name__ == '__main__':
    face_locations = []

    face_detector = FaceDetector()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        face_locations = face_detector.detect(frame)  # face detect

        # draw face box
        for face_loc in face_locations:
            (x, y, w, h) = face_loc
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('frame', frame)

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
