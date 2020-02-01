import pickle
import os
import cv2

PACKAGE_PATH = os.path.dirname(__file__)

DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
IMG_BASE_PATH = DATA_BASE_PATH + "/img_base/"

USER_TABLE = DATA_BASE_PATH + "/user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "/user_name_table.pk"
EMB_TABLE = DATA_BASE_PATH + "/emb_table.pk"

PARAMS_FILE = DATA_BASE_PATH + "/params.pk"
INDEX_FILE = DATA_BASE_PATH + "/index"


class DataBaseIO:
    def __init__(self):
        self.buffer = {}

    def load_user_name_table(self):
        return self.__load_table(USER_NAME_TABLE)

    def save_user_name_table(self, buffer_table):
        self.__save_table(USER_NAME_TABLE, buffer_table)

    def load_user_table(self):
        return self.__load_table(USER_TABLE)

    def save_user_table(self, buffer_table):
        self.__save_table(USER_TABLE, buffer_table)

    def load_emb_table(self):
        return self.__load_table(EMB_TABLE)

    def save_emb_table(self, buffer_table):
        self.__save_table(EMB_TABLE, buffer_table)

    def load_params(self):
        return self.__load_table(PARAMS_FILE)

    def save_params(self, params):
        self.__save_table(PARAMS_FILE, params)

    def load_img(self, file_name):
        img = cv2.imread(IMG_BASE_PATH + file_name)
        return img

    def save_img(self, file_name, frame):
        dir = file_name.rsplit("/", 1)

        if not os.path.isdir(IMG_BASE_PATH + dir[0]):
            os.mkdir(IMG_BASE_PATH + dir[0])

        cv2.imwrite(IMG_BASE_PATH + file_name, frame)

    def __load_table(self, table_file):
        print("loading table ...")
        if os.path.getsize(table_file) > 0:  # check file is not empty
            with open(table_file, "rb") as fp:
                buffer_table = pickle.load(fp)
        else:
            buffer_table = {}

        return buffer_table

    def __save_table(self, table_file, buffer_table):
        print("saving table ...")
        with open(table_file, "wb") as fp:
            pickle.dump(buffer_table, fp)
