from database_server.mongo_server import *
from pyflann import *
import numpy as np
import pickle

PACKAGE_PATH = os.path.dirname(__file__)
DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
PARAMS_FILE = DATA_BASE_PATH + "/params.pk"
INDEX_FILE = DATA_BASE_PATH + "/index"


TargetPrecision = 0.9
NumberNeighbors = 1
DistanceRate = 0.8


class FlannServer:
    def __init__(self):
        self.flann = None

        self.user_tb = UserTable()
        self.emb_tb = EmbTable()

        self.emb_tb_eids = None
        self.emb_tb_embs = None
        self.params = None

    def build(self):
        print("building flann...")

        # initial flann
        self.flann = FLANN()

        # load eids and embs
        self.emb_tb_eids, self.emb_tb_embs = self.emb_tb.get_eids_embs()
        self.emb_tb_embs = np.array(self.emb_tb_embs)

        if self.emb_tb_eids:

            # building flann index
            self.params = self.flann.build_index(self.emb_tb_embs, algorithm="autotuned",
                                                 target_precision=TargetPrecision)
            # save params and index
            self.__save_params(self.params)
            self.flann.save_index(INDEX_FILE.encode('utf-8'))

            return True

        else:
            return False

    def load(self):
        print("loading flann...")

        # initial flann
        self.flann = FLANN()

        # load eids and embs
        self.emb_tb_eids, self.emb_tb_embs = self.emb_tb.get_eids_embs()
        self.emb_tb_embs = np.array(self.emb_tb_embs)

        # load params and index
        self.params = self.__load_params()
        self.flann.load_index(INDEX_FILE.encode('utf-8'), self.emb_tb_embs)

    def search(self, inp_target_embs):
        print("searching target ...")

        target_embs = np.array(inp_target_embs)

        # index flann
        idxs, dists = self.flann.nn_index(target_embs,
                                          num_neighbors=NumberNeighbors,
                                          checks=self.params["checks"])
        # search user name
        user_names = []
        for idx, dist in zip(idxs, dists):
            if dist < DistanceRate:
                eid = self.emb_tb_eids[idx]
                uid = self.emb_tb.get_uid(eid)
                user_data = self.user_tb.get_user_data_by_uid(uid)
                user_names.append(user_data["name"])
            else:
                user_names.append("")

        return user_names

    def reset(self):
        # remove content from file
        files = [PARAMS_FILE, INDEX_FILE]
        for file in files:
            with open(file, "wb") as fp:
                fp.truncate()

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
    # remove all data
    flann_server = FlannServer()
    flann_server.reset()
