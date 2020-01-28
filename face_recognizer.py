import numpy as np
import cv2
import json
from skimage.transform import resize
from scipy.spatial import distance
from keras.models import load_model

# MS-Celeb-1M dataset pretrained Keras model
MODEL_PATH = "./model/facenet_keras.h5"


class FaceRecognizer:
    def __init__(self):
        self.model = load_model(MODEL_PATH)
        self.inp_frame = None
        self.face_locations = []
        self.raw_faces = []
        self.inp_faces = []
        self.face_embs = []
        self.face_dict = {}
        self.data_base_dict = {}

    def load_data_base(self, file_name):
        self.data_base_dict = {}

        with open(file_name, "r") as fp:
            load_dict = json.load(fp)

        for (ID, emb) in load_dict.items():
            self.data_base_dict[ID] = np.array(emb)

    # provide for face_trainer to face embedding
    def embedding(self, inp_frame, inp_face_locations):
        self.__read_frame(inp_frame, inp_face_locations)
        self.__extract_face()
        self.__preprocess_face()

        for (i, inp_face) in enumerate(self.inp_faces):
            face_vector = np.concatenate(self.model.predict(inp_face))
            self.face_embs.append(self.__l2_normalize(face_vector))

        return self.face_embs

    def recognize(self, inp_frame, inp_face_locations):
        self.__read_frame(inp_frame, inp_face_locations)
        self.__extract_face()
        self.__preprocess_face()
        self.__embedding_face()
        self.__recognize_face()
        self.__draw_face_box()

        return self.inp_frame

    def __read_frame(self, inp_frame, inp_face_locations):
        # clear params when read new frame
        self.inp_frame = inp_frame
        self.face_locations = inp_face_locations
        self.raw_faces = []
        self.inp_faces = []
        self.face_embs = []
        self.face_dict = {}

    def __extract_face(self):
        margin = 6
        for face_loc in self.face_locations:
            (x, y, w, h) = face_loc
            face = self.inp_frame[y:y + h, x:x + w]
            face_margin = np.zeros((h + margin * 2, w + margin * 2, 3), dtype="uint8")

            try:
                face_margin[margin:margin + h, margin:margin + w] = face
            except ValueError:
                # Camera shaking cause detect error location, and lead to extract error face shape
                print('traceback __face_extract func')
                print('{} : could not broadcast input array from shape {} into shape {}:'.format(ValueError, face.shape,
                                                                                                 face_margin.shape))

            self.raw_faces.append(face_margin)

    def __preprocess_face(self):
        image_size = 160  # facenet model need 160×160 image size

        for raw_face in self.raw_faces:
            face = resize(raw_face, (image_size, image_size), mode='reflect')  # resize face
            whiten_face = self.__pre_whiten(face)  # whiten face
            whiten_face = whiten_face[np.newaxis, :]
            self.inp_faces.append(whiten_face)

    def __embedding_face(self):
        for (i, inp_face) in enumerate(self.inp_faces):
            face_vector = np.concatenate(self.model.predict(inp_face))
            self.face_embs.append(self.__l2_normalize(face_vector))

    def __recognize_face(self):
        different_rate = 1

        for (i, face_emb) in enumerate(self.face_embs):
            min_dist = 1
            face_ID = 'unknown' + str(i)

            # search minimum distance face_emb and database emb
            for (ID, emb) in self.data_base_dict.items():
                dist = distance.euclidean(face_emb, emb)
                if dist < min_dist and dist < different_rate:
                    min_dist = dist
                    face_ID = ID

            self.face_dict[face_ID] = self.face_locations[i]

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
        for (face_ID, face_loc) in self.face_dict.items():
            (x, y, w, h) = face_loc
            cv2.rectangle(self.inp_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(self.inp_frame, face_ID, (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.putText(self.inp_frame, "Recognizing...", (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                        1, (0, 255, 0), 1, cv2.LINE_AA)
