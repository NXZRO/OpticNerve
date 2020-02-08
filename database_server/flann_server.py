from pyflann import *
import numpy as np
import pickle
from database_server.test_databse_server import DataBase

PACKAGE_PATH = os.path.dirname(__file__)
DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
PARAMS_FILE = DATA_BASE_PATH + "/params.pk"
INDEX_FILE = DATA_BASE_PATH + "/index"


TargetPrecision = 0.9
NumberNeighbors = 1
DistanceRate = 0.8


class FlannServer:
    def __init__(self):
        self.database = DataBase()
        self.flann = None

        self.database_embs = None
        self.params = None

        self.user_table = {}
        self.emb_table = {}
        self.emb_table_eids = []

    def build(self):
        print("building flann...")
        # load user table
        self.user_table = self.database.load_table("USER_TABLE")

        # initial flann
        self.flann = FLANN()

        # load embs
        self.emb_table = self.database.load_table("EMB_TABLE")
        self.emb_table_eids = list(self.emb_table.get_keys())
        embs = [emb_info["face_embs"] for emb_info in self.emb_table.get_values()]
        self.database_embs = np.array(embs)

        # building flann index
        self.params = self.flann.build_index(self.database_embs, algorithm="autotuned",
                                             target_precision=TargetPrecision)
        # save params and index
        self.__save_params(self.params)
        self.flann.save_index(INDEX_FILE.encode('utf-8'))

    def load(self):
        print("loading flann...")
        # load user table
        self.user_table = self.database.load_table("USER_TABLE")

        # initial flann
        self.flann = FLANN()

        # load embs
        self.emb_table = self.database.load_table("EMB_TABLE")
        self.emb_table_eids = list(self.emb_table.get_keys())
        embs = [emb_info["face_embs"] for emb_info in self.emb_table.get_values()]
        self.database_embs = np.array(embs)

        # load params and index
        self.params = self.__load_params()
        self.flann.load_index(INDEX_FILE.encode('utf-8'), self.database_embs)

    def search(self, inp_target_embs):
        print("searching target ...")

        target_embs = np.array(inp_target_embs)

        # index flann
        idxs, dists = self.flann.nn_index(target_embs,
                                          num_neighbors=NumberNeighbors,
                                          checks=self.params["checks"])
        # map user name from user table
        user_names = []
        for idx, dist in zip(idxs, dists):
            if dist < DistanceRate:
                eid = self.emb_table_eids[idx]
                uid = self.emb_table.get(eid)["uid"]
                user_names.append(self.user_table.get(uid)["name"])
            else:
                user_names.append("")

        return user_names

    def __load_params(self):
        if os.path.getsize(PARAMS_FILE) > 0:  # check file is not empty
            with open(PARAMS_FILE, "rb") as fp:
                buffer_table = pickle.load(fp)
        else:
            buffer_table = {}

        return buffer_table

    def __save_params(self, buffer_table):
        with open(PARAMS_FILE, "wb") as fp:
            pickle.dump(buffer_table, fp)


if __name__ == '__main__':
    db = DataBase()

    user_name_table = db.load_table("USER_NAME_TABLE")
    uid = user_name_table.get("Kp")

    user_table = db.load_table("USER_TABLE")
    user_info = user_table.get(uid)
    embs = user_info["face_embs"]

    flann_server = FlannServer()
    flann_server.build()
    flann_server.load()
    names = flann_server.search(embs[0])
    print(names)
