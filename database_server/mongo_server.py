from pymongo import MongoClient

USER_TABLE = "user_tb"
EMB_TABLE = "emb_tb"
ID_TABLE = "id_tb"


class MongoServer:
    def __init__(self):
        self.__client = MongoClient("localhost", 27017)
        self.db = self.__client.face_db
        self.__id_tb = self.db[ID_TABLE]
        self.__user_tb = self.db[USER_TABLE]
        self.__emb_tb = self.db[EMB_TABLE]

    def reset(self):
        self.__id_tb.remove({})
        self.__user_tb.remove({})
        self.__emb_tb.remove({})

        initial_id_state = {"uid": 0, "eid": 0}
        self.__id_tb.insert_one(initial_id_state)


class IdTable(MongoServer):
    def __init__(self):
        super().__init__()
        self.__id_tb = self.db[ID_TABLE]
        self.__uid = None
        self.__eid = None

    def update_uid(self, new_uid):
        self.__get_ids()
        self.__id_tb.update({"uid": self.__uid}, {"$set": {"uid": new_uid}})

    def update_eid(self, new_eid):
        self.__get_ids()
        self.__id_tb.update({"eid": self.__eid}, {"$set": {"eid": new_eid}})

    def get_uid(self):
        self.__get_ids()
        return self.__uid

    def get_eid(self):
        self.__get_ids()
        return self.__eid

    def __get_ids(self):
        for data in self.__id_tb.find():
            self.__uid = data["uid"]
            self.__eid = data["eid"]

    def show_table(self):
        for tb in self.__id_tb.find():
            print(tb)


class UserTable(MongoServer):
    def __init__(self):
        super().__init__()
        self.__user_tb = self.db[USER_TABLE]

    def insert_user(self, data):
        self.__user_tb.insert_one(data)

    def get_user_data_by_name(self, name):
        user_data = None
        for data in self.__user_tb.find({"name": name}):
            user_data = data
        return user_data

    def get_user_data_by_uid(self, uid):
        user_data = None
        for data in self.__user_tb.find({"uid": uid}):
            user_data = data
        return user_data

    def remove_user(self, uid):
        self.__user_tb.remove({"uid": uid})

    def show_table(self):
        for tb in self.__user_tb.find():
            print(tb)


class EmbTable(MongoServer):
    def __init__(self):
        super().__init__()
        self.__emb_tb = self.db[EMB_TABLE]

    def insert_emb(self, data):
        self.__emb_tb.insert_one(data)

    def get_emb_data(self, eid):
        emb_data = None
        for data in self.__emb_tb.find({"eid": eid}):
            emb_data = data
        return emb_data

    def get_uid(self, eid):
        uid = None
        for data in self.__emb_tb.find({"eid": eid}):
            uid = data["uid"]
        return uid

    def get_eids_embs(self):
        eids = []
        embs = []

        for emb_data in self.__emb_tb.find():
            eids.append(emb_data["eid"])
            embs.append(emb_data["face_emb"])

        return eids, embs

    def remove_emb(self, eid):
        self.__emb_tb.remove({"eid": eid})

    def show_table(self):
        for tb in self.__emb_tb.find():
            print(tb)


if __name__ == '__main__':
    # remove all data
    db = MongoServer()
    db.reset()

    # initial table
    id_tb = IdTable()
    emb_tb = EmbTable()
    user_tb = UserTable()

    # show tables
    id_tb.show_table()
    user_tb.show_table()
    # emb_tb.show_table()
