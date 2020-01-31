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
            self.face_locations.append(tuple(face['box']))

        return self.face_locations

    def __read_frame(self, inp_frame):
        # clear params when read new frame
        self.inp_frame = inp_frame
        self.face_locations = []
