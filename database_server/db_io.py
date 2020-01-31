import pickle
import os
import cv2

PACKAGE_PATH = os.path.dirname(__file__)

DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
IMG_BASE_PATH = DATA_BASE_PATH + "/img_base/"
CHECK_IMG_BASE_PATH = DATA_BASE_PATH + "/check_img_base/"

ID_FILE = DATA_BASE_PATH + "/ID.txt"
USER_TABLE = DATA_BASE_PATH + "/user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "/user_name_table.pk"

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

    def load_params(self):
        return self.__load_table(PARAMS_FILE)

    def save_params(self, params):
        self.__save_table(PARAMS_FILE, params)

    def load_img(self, file_name):
        img = cv2.imread(IMG_BASE_PATH + file_name)
        return img

    def save_img(self, file_name, frame):
        cv2.imwrite(IMG_BASE_PATH + file_name, frame)

    def load_check_img(self, file_name):
        img = cv2.imread(CHECK_IMG_BASE_PATH + file_name)
        return img

    def save_check_img(self, file_name, frame):
        cv2.imwrite(CHECK_IMG_BASE_PATH + file_name, frame)

    def read_ID(self, user_names):
        with open(ID_FILE, "r") as fp:
            for id in fp:
                user_names.append(str(id).rstrip('\n'))
        return user_names

    def get_img_base_files_path(self):
        return os.listdir(IMG_BASE_PATH)

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
