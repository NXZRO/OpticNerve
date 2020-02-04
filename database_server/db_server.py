from pyflann import *
import numpy as np
from database_server.db_io import DataBaseIO

PACKAGE_PATH = os.path.dirname(__file__)
DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
INDEX_FILE = DATA_BASE_PATH + "/index"

TargetPrecision = 0.9
NumberNeighbors = 1

DistanceRate = 0.8


class DataBaseServer:
    def __init__(self):
        self.io = DataBaseIO()
        self.flann = None
        self.user_table = {}
        self.emb_table = {}
        self.database_embs = []
        self.target_embs = []
        self.eids = []
        self.emb_dists = []
        self.user_names = []

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
        self.__search_user_name()

        return self.user_names

    def __load_database_embs(self):
        self.emb_table = self.io.load_emb_table()
        embs = [emb_info["face_embs"] for emb_info in self.emb_table.values()]
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
        self.eids = idxs
        self.emb_dists = dists

    def __search_user_name(self):
        self.user_table = self.io.load_user_table()
        self.user_names = []
        i = 0
        for eid, emb_dist in zip(self.eids, self.emb_dists):
            if emb_dist < DistanceRate:
                uid = self.emb_table[eid]["uid"]
                self.user_names.append(self.user_table[uid]["name"])
            else:
                self.user_names.append("Unknow" + str(i))
                i += 1

    def __load_idx(self):
        self.params = self.io.load_params()
        self.flann.load_index(INDEX_FILE.encode('utf-8'), self.database_embs)

    def __save_idx(self):
        self.io.save_params(self.params)
        self.flann.save_index(INDEX_FILE.encode('utf-8'))


if __name__ == '__main__':

    server = DataBaseServer()
    user_name_table = server.io.load_user_name_table()
    print(user_name_table)
    uid = user_name_table["Kp"]
    user_table = server.io.load_user_table()
    user_info = user_table[uid]
    target_embs = user_info["face_embs"]
    print(target_embs)
    op = 0
    if op == 0:
        server.build_database()
    else:
        server.load_database()

    user_names = server.search_database(target_embs[0])
    print(user_names)

