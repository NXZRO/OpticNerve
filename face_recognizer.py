import numpy as np
import cv2
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model
from mtcnn import MTCNN

# MS-Celeb-1M dataset pretrained Keras model
MODEL_PATH = "./model/facenet_keras.h5"

DATA_BASE_PATH = "./data_base/"


class FaceRecognizer:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.mtcnn_detector = MTCNN()
        self.inp_frame = None
        self.face_positions = []
        self.raw_faces = []
        self.inp_faces = []
        self.data_base_dict = {}
        self.face_dict = {}

    def training_data(self):
        # read ID and set ID as dict key
        file = open(DATA_BASE_PATH + "ID.txt", "r")
        for ID in file:
            self.data_base_dict[str(ID).rstrip('\n')] = None

        # compute emb and set emb as dict value
        for (i, ID) in enumerate(self.data_base_dict.keys()):
            img = cv2.imread(DATA_BASE_PATH + str(i) + '.jpg')
            self.__read_frame(img)
            self.__face_detect()
            self.__face_extract()
            self.__face_preprocess()
            face_vector = np.concatenate(self.model.predict(self.inp_faces[0]))
            face_emb = self.__l2_normalize(face_vector)
            self.data_base_dict[ID] = face_emb

    def recognize(self, inp_frame):
        face_recognizer.__read_frame(inp_frame)
        face_recognizer.__face_detect()
        face_recognizer.__face_extract()
        face_recognizer.__face_preprocess()
        face_recognizer.__face_recognize()
        face_recognizer.__draw_face_box()
        return self.inp_frame

    def __read_frame(self, inp_frame):
        # clear params when read new frame
        self.inp_frame = inp_frame
        self.face_positions = []
        self.raw_faces = []
        self.inp_faces = []
        self.face_dict = {}

    def __face_detect(self):
        faces = self.mtcnn_detector.detect_faces(self.inp_frame)

        for face in faces:
            self.face_positions.append(face['box'])

    def __face_extract(self):
        margin = 6
        for face_pos in self.face_positions:
            (x, y, w, h) = face_pos
            face = self.inp_frame[y:y + h, x:x + w]
            face_margin = np.zeros((h + margin * 2, w + margin * 2, 3), dtype="uint8")

            try:
                face_margin[margin:margin + h, margin:margin + w] = face
            except ValueError:
                # Camera shaking cause detect error position, and lead to extract error face shape
                print('traceback __face_extract func')
                print('{} : could not broadcast input array from shape {} into shape {}:'.format(ValueError, face.shape,
                                                                                                 face_margin.shape))

            self.raw_faces.append(face_margin)

    def __face_preprocess(self):
        image_size = 160  # facenet model need 160×160 image size

        for raw_face in self.raw_faces:
            face = resize(raw_face, (image_size, image_size), mode='reflect')  # resize face
            whiten_face = self.__pre_whiten(face)  # whiten face
            whiten_face = whiten_face[np.newaxis, :]
            self.inp_faces.append(whiten_face)

    def __face_recognize(self):
        different_rate = 1

        for (i, inp_face) in enumerate(self.inp_faces):
            face_vector = np.concatenate(self.model.predict(inp_face))
            face_emb = self.__l2_normalize(face_vector)

            min_dist = 1
            face_ID = 'unknown' + str(i)

            # search minimum distance face_emb and database emb
            for (ID, emb) in self.data_base_dict.items():
                dist = distance.euclidean(face_emb, emb)
                if dist < min_dist and dist < different_rate:
                    min_dist = dist
                    face_ID = ID

            self.face_dict[face_ID] = self.face_positions[i]

    def __pre_whiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size

        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size

        else:
            raise ValueError("Dimension should be 3 or 4")

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj

        return y

    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

        return output

    def __draw_face_box(self):
        for (face_ID, face_pos) in self.face_dict.items():
            (x, y, w, h) = face_pos
            cv2.rectangle(self.inp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.inp_frame, face_ID, (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)


if __name__ == '__main__':

    face_recognizer = FaceRecognizer()

    face_recognizer.training_data()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:
        ret, frame = camera.read()  # get frame

        frame = face_recognizer.recognize(frame)  # recognize frame

        cv2.imshow('frame', frame)

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
