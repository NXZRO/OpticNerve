from pyflann import *
import numpy as np
import pickle


DATA_BASE_PATH = "./data_base/"
IMG_BASE_PATH = DATA_BASE_PATH + "img_base/"
CHECK_IMG_BASE_PATH = DATA_BASE_PATH + "check_img_base/"

ID_FILE = DATA_BASE_PATH + "ID.txt"
USER_TABLE = DATA_BASE_PATH + "user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "user_name_table.pk"
EMB_TABLE = DATA_BASE_PATH + "emb_table.pk"

INDEX_FILE = DATA_BASE_PATH + "index"
PARAMS_FILE = DATA_BASE_PATH + "params.pk"

TargetPrecision = 0.9
NumberNeighbors = 1

DistanceRate = 1


class DataBaseServer:
    def __init__(self):
        self.flann = None
        self.user_table = {}
        self.database_embs = None
        self.target_embs = None
        self.uids = None
        self.emb_dists = None
        self.user_name = []

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

    def build_database(self):
        self.flann = FLANN()
        self.__load_database_embs()
        self.__build_idx()
        self.__save_idx()

    def load_database(self):
        self.flann = FLANN()
        self.__load_database_embs()
        self.__load_idx()

    def search_database(self, target_embs):
        self.target_embs = np.array(target_embs)
        self.__search_idx()

        self.user_name = []
        i = 0
        for uid, emb_dist in zip(self.uids, self.emb_dists):
            if emb_dist < DistanceRate:
                self.user_name.append(self.user_table[uid][0])
            else:
                self.user_name.append("Unknow" + str(i))
                i += 1

        return self.user_name

    def __load_database_embs(self):
        self.user_table = self.load_table(USER_TABLE)
        embs = [user_info[1] for user_info in self.user_table.values()]
        self.database_embs = np.array(embs)

    def __build_idx(self):
        print("start building ...")
        self.params = self.flann.build_index(self.database_embs, algorithm="autotuned",
                                             target_precision=TargetPrecision)

    def __search_idx(self):
        print("searching target ...")
        idxs, dists = self.flann.nn_index(self.target_embs,
                                          num_neighbors=NumberNeighbors,
                                          checks=self.params["checks"])
        self.uids = idxs
        self.emb_dists = dists

    def __load_idx(self):
        print("loading index ...")
        with open(PARAMS_FILE, 'rb') as fp:
            self.params = pickle.load(fp)
        self.flann.load_index(INDEX_FILE.encode('utf-8'), self.database_embs)

    def __save_idx(self):
        print("saving index ...")
        with open(PARAMS_FILE, 'wb') as fp:
            pickle.dump(self.params, fp)
        self.flann.save_index(INDEX_FILE.encode('utf-8'))


if __name__ == '__main__':

    server = DataBaseServer()
    user_name_table = server.load_table(USER_NAME_TABLE)
    print(user_name_table)
    uid = user_name_table["Obama"]
    user_table = server.load_table(USER_TABLE)
    user_info = user_table[uid]
    target_emb = user_info[1]

    op = 1
    if op == 0:
        server.build_database()
    else:
        server.load_database()

    user_names = server.search_database([target_emb])
    print(user_names[0])
