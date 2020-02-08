import pickle
import os
import cv2
import shutil

PACKAGE_PATH = os.path.dirname(__file__)

DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
IMG_BASE_PATH = DATA_BASE_PATH + "/img_base/"

USER_TABLE = DATA_BASE_PATH + "/user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "/user_name_table.pk"
EMB_TABLE = DATA_BASE_PATH + "/emb_table.pk"

PARAMS_FILE = DATA_BASE_PATH + "/params.pk"
INDEX_FILE = DATA_BASE_PATH + "/index"


class Table:
    def __init__(self, table_name):
        self.table_name = table_name
        self.__table = None

    def new(self, key, value):
        self.__table = self.__load()
        self.__table[key] = value
        self.__save(self.__table)

    def delete(self, key):
        self.__table = self.__load()
        self.__table.pop(key)
        self.__save(self.__table)

    def get(self, key):
        self.__table = self.__load()
        return self.__table.setdefault(key, None)

    def get_keys(self):
        self.__table = self.__load()
        return self.__table.keys()

    def get_values(self):
        self.__table = self.__load()
        return self.__table.values()

    def show(self):
        table = self.__load()
        print(table)

    def __load(self):
        if os.path.getsize(self.table_name) > 0:  # check file is not empty
            with open(self.table_name, "rb") as fp:
                buffer_table = pickle.load(fp)
        else:
            buffer_table = {}

        return buffer_table

    def __save(self, buffer_table):
        with open(self.table_name, "wb") as fp:
            pickle.dump(buffer_table, fp)

class DataBase:
    def __init__(self):
        self.__TableName = {"USER_TABLE": USER_TABLE, "USER_NAME_TABLE": USER_NAME_TABLE, "EMB_TABLE": EMB_TABLE}
        self.table = None

    def load_table(self, table_name):
        return Table(self.__TableName[table_name])

    def load_img(self, file_name):
        img = cv2.imread(IMG_BASE_PATH + file_name)
        return img

    def save_img(self, file_name, frame):
        dir = file_name.rsplit("/", 1)

        if not os.path.isdir(IMG_BASE_PATH + dir[0]):
            os.mkdir(IMG_BASE_PATH + dir[0])

        cv2.imwrite(IMG_BASE_PATH + file_name, frame)

    def remove_img(self, file_name):
        shutil.rmtree(IMG_BASE_PATH + file_name)  # delete user img dir

    def clear(self):
        tables = [USER_TABLE,USER_NAME_TABLE,EMB_TABLE,PARAMS_FILE,INDEX_FILE]
        # delete table
        for table_file in tables:
            with open(table_file, "wb") as fp:
                fp.truncate()

        # delete img base dir
        shutil.rmtree(IMG_BASE_PATH)

        # create new img dir
        os.mkdir(IMG_BASE_PATH)


if __name__ == '__main__':
    db = DataBase()
    db.clear()
    user_table = db.load_table("USER_NAME_TABLE")
    user_table.show()
