import cv2
import json
from face_recognizer import FaceRecognizer
from face_detector import FaceDetector

DATA_BASE_PATH = "./data_base/"


class FaceTrainer:

    def __init__(self):
        self.face_detector = FaceDetector()
        self.face_recognizer = FaceRecognizer()
        self.data_base_dict = {}
        self.face_locations = []
        self.face_embs = []

    def training_data(self):
        # read ID and set ID as dict key
        file = open(DATA_BASE_PATH + "ID.txt", "r")
        for ID in file:
            self.data_base_dict[str(ID).rstrip('\n')] = None

        # compute emb and set emb as dict value
        for (i, ID) in enumerate(self.data_base_dict.keys()):
            img = cv2.imread(DATA_BASE_PATH + str(i) + ".jpg")              # read img
            face_locations = self.face_detector.detect(img)                 # detect_face
            face_embs = self.face_recognizer.training(img, face_locations)  # training (get face embeds)
            self.data_base_dict[ID] = face_embs[0]                          # save into dict

        # write data_base_dict back database
        self.__write_data()

    def __write_data(self):
        out_dict = {}
        for (ID, emb) in self.data_base_dict.items():
            out_dict[ID] = emb.tolist()

        with open(DATA_BASE_PATH + "Data.json", "w") as fp:
            fp.write(json.dumps(out_dict))
