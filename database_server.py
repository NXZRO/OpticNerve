from pyflann import *
import numpy as np
import json
import pickle

index_file = "data_base/index"
data_file = "data_base/Data.json"
params_file = "data_base/params.pk"

class DataBaseServer:
    def __init__(self):
        self.flann = None
        self.dataset = None
        self.target = None
        self.params = None
        self.match_dataset_idx = None
        self.match_dataset_dists = None

    def build_data_base(self):
        self.__initial_flann()
        self.__initial_dataset()
        self.__build_idx()
        self.__save_idx()

    def load_data_base(self):
        self.__initial_flann()
        self.__initial_dataset()
        self.__load_idx()

    def search_data_base(self, target):
        self.target = target
        self.__search_idx()

    def __initial_dataset(self):
        data_base_dict = {}

        with open(data_file, "r") as fp:
            load_dict = json.load(fp)
            for (ID, emb) in load_dict.items():
                data_base_dict[ID] = np.array(emb)

        t_arr = np.array([data_base_dict["Obama"]])
        a_arr = np.random.rand(100, 128)
        self.dataset = np.append(a_arr, t_arr, axis=0)

    def __initial_flann(self):
        self.flann = FLANN()

    def __build_idx(self):
        print("start building")
        self.params = self.flann.build_index(self.dataset, algorithm="autotuned", target_precision=0.9)

    def __save_idx(self):
        print("save idx")
        pickle.dump(self.params, open(params_file, 'wb'))
        self.flann.save_index(index_file.encode('utf-8'))

    def __load_idx(self):
        self.params = pickle.load(open(params_file, 'rb'))
        self.flann.load_index(index_file.encode('utf-8'), self.dataset)

    def __search_idx(self):
        self.match_dataset_idx, self.match_dataset_dists = self.flann.nn_index(self.target, num_neighbors=1,
                                                                          checks=self.params["checks"])


if __name__ == '__main__':
    data_base_dict = {}
    with open(data_file, "r") as fp:
        load_dict = json.load(fp)
        for (ID, emb) in load_dict.items():
            data_base_dict[ID] = np.array(emb)

    target = np.array([data_base_dict["Obama"]])

    server = DataBaseServer()
    server.load_data_base()
    server.search_data_base(target)

    print(server.match_dataset_idx)
    print(server.dataset[server.match_dataset_idx])
    print(target)
