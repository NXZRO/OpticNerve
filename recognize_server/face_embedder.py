from facenet_pytorch import InceptionResnetV1
import numpy as np
import torch
import cv2
import time


class FaceEmbedder:

    def __init__(self):
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = InceptionResnetV1(pretrained='vggface2').eval().to(self.device)

    def embedding(self, frame, face_locations):

        raw_face = self.__extract_faces(frame, face_locations)

        whiten_face = self.__pre_whiten(raw_face)

        inp_face = self.__transform_inp_faces(whiten_face)

        predicted_embeddings = self.__embedding_faces(inp_face)

        face_embeddings = self.__l2_normalize(predicted_embeddings)

        return face_embeddings

    def __extract_faces(self, frame, face_locations):
        faces = []
        for face_loc in face_locations:
            (x1, y1, x2, y2) = face_loc

            face = frame[y1:y2, x1:x2]

            face = cv2.resize(face, (160, 160))

            face = np.array(face)

            faces.append(face)

        return faces

    def __pre_whiten(self, faces):
        axis = (1, 2, 3)
        size = faces[0].size

        mean = np.mean(faces, axis=axis, keepdims=True)
        std = np.std(faces, axis=axis, keepdims=True)
        std_adj = np.maximum(std, 1.0 / np.sqrt(size))
        faces = (faces - mean) / std_adj

        return faces

    def __transform_inp_faces(self, faces):

        faces = np.transpose(faces, (0, 3, 1, 2))

        faces = torch.tensor(faces)

        faces = faces.type(torch.FloatTensor)

        return faces

    def __embedding_faces(self, faces):

        faces = torch.stack(tuple(face for face in faces)).to(self.device)

        predicted_embeddings = self.model(faces).detach().cpu()

        return predicted_embeddings

    def __l2_normalize(self, x, axis=-1, epsilon=1e-10):
        x = x.numpy().astype('float64')
        y = x / np.sqrt(np.maximum(np.sum(np.square(x), axis=axis, keepdims=True), epsilon))

        return y


