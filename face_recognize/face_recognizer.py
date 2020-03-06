import numpy as np
from skimage.transform import resize
import tensorflow as tf


import os


PACKAGE_PATH = os.path.dirname(__file__)

# MS-Celeb-1M dataset pretrained Keras model
MODEL_PATH = PACKAGE_PATH + "/model/facenet_keras.h5"


class FaceRecognizer:

    def __init__(self):
        self.model = tf.keras.models.load_model(MODEL_PATH)
        self.inp_frame = None
        self.face_locations = []
        self.raw_faces = []
        self.inp_faces = []
        self.face_embs = []

    def recognize(self, inp_frame, inp_face_locations):
        self.__read_frame(inp_frame, inp_face_locations)
        self.__extract_face()
        if self.raw_faces is None:
            return None
        else:
            self.__preprocess_face()
            self.__embedding_face()

            return self.face_embs

    def __read_frame(self, inp_frame, inp_face_locations):
        # clear params when read new frame
        self.inp_frame = inp_frame
        self.face_locations = inp_face_locations
        self.raw_faces = []
        self.inp_faces = []
        self.face_embs = []

    def __extract_face(self):
        margin = 6
        for face_loc in self.face_locations:
            (x, y, w, h) = face_loc
            face = self.inp_frame[y:y + h, x:x + w]
            face_margin = np.zeros((h + margin * 2, w + margin * 2, 3), dtype="uint8")

            face_margin[margin:margin + h, margin:margin + w] = face

            self.raw_faces.append(face_margin)

            # try:
            #     face_margin[margin:margin + h, margin:margin + w] = face
            # except ValueError:
            #     # Camera shaking cause detect error location, and lead to extract error face shape
            #     print('traceback __face_extract func')
            #     print('{} : could not broadcast input array from shape {} into shape {}:'.format(ValueError, face.shape,
            #                                                                                      face_margin.shape))
            #     self.raw_faces = None
            #     break
            # else:
            #     self.raw_faces.append(face_margin)

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
            face_emb = self.__l2_normalize(face_vector)
            self.face_embs.append(face_emb.astype('float64'))

    def __pre_whiten(self, x):
        if x.ndim == 4:
            axis = (1, 2, 3)
            size = x[0].size

        elif x.ndim == 3:
            axis = (0, 1, 2)
            size = x.size

        else:
            raise ValueError("Dimension should be 3 or 1")

        mean = np.mean(x, axis=axis, keepdims=True)
        std = np.std(x, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        y = (x - mean) / std_adj

        return y

    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        output = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

        return output
