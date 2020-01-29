from face_recognizer import FaceRecognizer
from face_detector import FaceDetector
import cv2
import pickle
import os


DATA_BASE_PATH = "./data_base/"
IMG_BASE_PATH = DATA_BASE_PATH + "img_base/"
CHECK_IMG_BASE_PATH = DATA_BASE_PATH + "check_img_base/"

ID_FILE = DATA_BASE_PATH + "ID.txt"
USER_TABLE = DATA_BASE_PATH + "user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "user_name_table.pk"


class FaceTrainer:

    def __init__(self):
        self.user_name = ""
        self.user_face_emb = None

        self.user_name_table = {}
        self.uid = 0

        self.user_table = {}
        self.user_info = []

    def new_user(self, user_name, user_face_embs):
        self.user_name = user_name
        self.user_face_emb = user_face_embs[0]
        self.__save_user()

    def __save_user(self):
        # save into user name table
        self.user_name_table = self.load_table(USER_NAME_TABLE)
        self.uid = self.user_name_table.setdefault(self.user_name, len(self.user_name_table))
        self.save_table(USER_NAME_TABLE, self.user_name_table)

        # save into user table
        self.user_table = self.load_table(USER_TABLE)
        self.user_info = [self.user_name, self.user_face_emb]
        self.user_table.setdefault(self.uid, self.user_info)
        self.save_table(USER_TABLE, self.user_table)

    def load_table(self, table_file):
        print("loading table ...")
        if os.path.getsize(table_file) > 0:  # check file is not empty
            with open(table_file, "rb") as fp:
                buffer_table = pickle.load(fp)
        else:
            buffer_table = {}

        return buffer_table

    def save_table(self, table_file, buffer_table):
        print("saving table ...")
        with open(table_file, "wb") as fp:
            pickle.dump(buffer_table, fp)


def draw_face(frame, face_locations, user_name):
    color = (255, 0, 0)
    for face_loc in face_locations:
        (x, y, w, h) = face_loc
        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.imwrite(CHECK_IMG_BASE_PATH + user_name + ".jpg", frame)


def read_ID(user_names):
    with open(ID_FILE, "r") as fp:
        for id in fp:
            user_names.append(str(id).rstrip('\n'))
    return user_names


if __name__ == '__main__':
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    trainer = FaceTrainer()

    # read data and new users
    # user_names = []
    # user_names = read_ID(user_names)
    #
    # img_files = os.listdir(IMG_BASE_PATH)
    #
    # for user_name, img_file in zip(user_names, img_files):
    #     img = cv2.imread(IMG_BASE_PATH + img_file)  # read img
    #     face_locs = detector.detect(img)
    #     user_face_embs = recognizer.recognize(img, face_locs)
    #     draw_face(img, face_locs, user_name)
    #     trainer.new_user(user_name, user_face_embs)
    #
    # # show table
    # user_table = trainer.load_table(USER_TABLE)
    # print(user_table)
    #
    # user_name_table = trainer.load_table(USER_NAME_TABLE)
    # print(user_name_table)

    # ---------------------------------------------------
    # new one user
    img = cv2.imread(IMG_BASE_PATH + "0.jpg")  # read img
    face_locs = detector.detect(img)
    user_face_embs = recognizer.recognize(img, face_locs)
    draw_face(img, face_locs, "Obama")
    trainer.new_user("Obama", user_face_embs)

    # show table
    user_table = trainer.load_table(USER_TABLE)
    print(user_table)

    user_name_table = trainer.load_table(USER_NAME_TABLE)
    print(user_name_table)



