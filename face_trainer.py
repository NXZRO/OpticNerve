from face_recognizer import FaceRecognizer
from face_detector import FaceDetector
import cv2
import pickle
from os import listdir
import numpy as np

DATA_BASE_PATH = "./data_base/"
IMG_BASE_PATH = DATA_BASE_PATH + "img_base/"
DATA_SET_FILE = DATA_BASE_PATH + "Data.pk"


class FaceTrainer:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.data_base_dict = {}
        self.face_locations = []
        self.face_embs = []

    def training_data(self):
        img_files = listdir(IMG_BASE_PATH)

        for ID, img_file in enumerate(img_files):
            img = cv2.imread(IMG_BASE_PATH + img_file)                       # read img
            face_locations = self.face_detector.detect(img)                  # detect_face
            face_embs = self.face_recognizer.embedding(img, face_locations)  # face embedding
            self.data_base_dict[ID] = face_embs[0]                           # save into dict

        self.__save_data_base()  # write data_base_dict to database

    def loading_data_base(self):
        self.__load_data_base()

    def __save_data_base(self):
        out_dict = {}
        for (ID, emb) in self.data_base_dict.items():
            out_dict[ID] = emb.tolist()

        with open(DATA_SET_FILE, "wb") as fp:
            pickle.dump(out_dict, fp)

    def __load_data_base(self):
        self.data_base_dict = {}
        with open(DATA_SET_FILE, 'rb') as fp:
                load_dict = pickle.load(fp)

        for (ID, emb) in load_dict.items():
            self.data_base_dict[ID] = np.array(emb)


if __name__ == '__main__':
    trainer = FaceTrainer()
    trainer.training_data()
    trainer.loading_data_base()
    print(trainer.data_base_dict[7])
