from mtcnn import MTCNN
import cv2


class FaceDetector:
    def __init__(self):
        self.mtcnn_detector = MTCNN()
        self.inp_frame = None
        self.face_locations = []

    def detect(self, inp_frame):
        self.__read_frame(inp_frame)
        self.__transform_frame_rgb()
        faces = self.mtcnn_detector.detect_faces(self.inp_frame)

        for face in faces:
            face_loc = face['box']
            print(face_loc)
            if face_loc:
                face_loc = self.__check_location(face_loc, self.inp_frame.shape)
                self.face_locations.append(face_loc)

        return self.face_locations

    def __read_frame(self, inp_frame):
        """ clear attribute, when read new frame """
        self.inp_frame = inp_frame
        self.face_locations = []

    def __transform_frame_rgb(self):
        """ transform bgr to rgb frame """
        b, g, r = cv2.split(self.inp_frame)
        self.inp_frame = cv2.merge([r, g, b])

    def __check_location(self, face_loc, frame_shape):
        """ reset face box location, when out of frame """
        (max_h, max_w, c) = frame_shape
        (x, y, w, h) = face_loc
        x = self.__naturalize_number(x)
        y = self.__naturalize_number(y)
        w = self.__reset_boundary(x, w, max_w)
        h = self.__reset_boundary(y, h, max_h)
        face_loc = tuple((x, y, w, h))
        return face_loc

    @staticmethod
    def __reset_boundary(x, dx, max_x):
        if (x + dx) > max_x:
            return max_x - x
        else:
            return dx

    @staticmethod
    def __naturalize_number(x):
        return 0 if x < 0 else x

