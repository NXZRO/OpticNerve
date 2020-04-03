from database_server.mongo_server import *
from database_server.img_server import *
from database_server.flann_server import *
import numpy as np


def new_user(user_name, face_embs):

    user_data = user_tb.get_user_data_by_name(user_name)

    if user_data is not None:
        print("user '{}' is exist ...".format(user_name))

    else:
        print("new user '{}' ...".format(user_name))

        # new uid and eid, then update uid and eid to user table
        curr_uid = id_tb.get_uid()
        curr_eid = id_tb.get_eid()

        uid = curr_uid + 1
        eids = [curr_eid + i + 1 for i, _ in enumerate(face_embs)]

        id_tb.update_uid(uid)
        id_tb.update_eid(eids[-1])

        # insert user to user table
        user_data = {"uid": uid, "name": user_name, "eids": eids}
        user_tb.insert_user(user_data)

        # insert face embs to emb table
        for eid, emb in zip(eids, face_embs):
            emb_data = {"eid": eid, "face_emb": emb.tolist(), "uid": uid}
            emb_tb.insert_emb(emb_data)

        print("new user success ...")


def remove_user(user_name):

    user_data = user_tb.get_user_data_by_name(user_name)

    if user_data is None:
        print("user '{}' isn't exist ...".format(user_name))

    else:
        print("remove user '{}' ...".format(user_name))

        # remove user from user table
        user_tb.remove_user(user_data["uid"])

        # remove face embs from emb table
        for eid in user_data["eids"]:
            emb_tb.remove_emb(eid)

        print("remove user success ...")


def clear_database():
    db = MongoServer()
    db.reset()

    img_server = ImgServer()
    img_server.reset()

    flann_server = FlannServer()
    flann_server.reset()


if __name__ == '__main__':
    db = MongoServer()
    # clear_database()

    # initial table
    id_tb = IdTable()
    emb_tb = EmbTable()
    user_tb = UserTable()

    # show tables
    id_tb.show_table()
    user_tb.show_table()
    # emb_tb.show_table()
