from facenet_pytorch import MTCNN
import torch
import cv2
import time


class FaceDetector:
    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn_detector = MTCNN(keep_all=True, device=self.device)

    def detect(self, frame):
        face_locations = []

        inp_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        boxes, probs = self.mtcnn_detector.detect(inp_frame)

        if boxes is not None:
            for box, prob in zip(boxes, probs):
                face_loc = self.__check_location(box.astype(int).tolist(), inp_frame.shape)
                face_locations.append(face_loc)

        return face_locations

    def __check_location(self, face_loc, frame_shape):
        """ reset face box location, when out of frame """
        (max_h, max_w, c) = frame_shape
        (x1, y1, x2, y2) = face_loc

        x1 = self.__check_positive(x1)
        y1 = self.__check_positive(y1)
        x2 = self.__check_border(x2, max_w)
        y2 = self.__check_border(y2, max_h)
        face_loc = tuple((x1, y1, x2, y2))

        return face_loc

    @staticmethod
    def __check_border(x2, max_x):
        if x2 > max_x:
            return max_x
        else:
            return x2

    @staticmethod
    def __check_positive(x):
        return 0 if x < 0 else x


if __name__ == '__main__':

    detector = FaceDetector()

    camera = cv2.VideoCapture(0)  # Fish -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        t1 = time.clock()

        face_locs = detector.detect(frame)  # detect face

        # draw face
        if face_locs:
            for face_loc in face_locs:
                (x1, y1, x2, y2) = face_loc
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        t2 = time.clock()
        print("time:", t2-t1)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
