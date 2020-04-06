from database_server.mongo_server import UserTable, EmbTable, IdTable, LogTable
from database_server.img_server import ImgServer


class UserServer:

    def __init__(self):
        self.id_tb = IdTable()
        self.user_tb = UserTable()
        self.emb_tb = EmbTable()
        self.log_tb = LogTable()
        self.img_server = ImgServer()

    def new_user(self, user_name, user_face_embs, user_face_imgs):
        user_data = self.user_tb.get_user_data_by_name(user_name)

        if user_data is not None:
            print("user '{}' is exist ...".format(user_name))
            return False

        else:
            print("new user '{}' ...".format(user_name))

            # new uid and eid, then update uid and eid to user table
            curr_uid = self.id_tb.get_uid()
            curr_eid = self.id_tb.get_eid()

            uid = curr_uid + 1
            eids = [curr_eid + i + 1 for i, _ in enumerate(user_face_embs)]

            self.id_tb.update_uid(uid)
            self.id_tb.update_eid(eids[-1])

            # insert user to user table
            user_data = {"uid": uid, "name": user_name, "eids": eids}
            self.user_tb.insert_user(user_data)

            # insert face embs to emb table
            for eid, emb in zip(eids, user_face_embs):
                emb_data = {"eid": eid, "face_emb": emb.tolist(), "uid": uid}
                self.emb_tb.insert_emb(emb_data)

            # save face img to img base
            for i, img in enumerate(user_face_imgs):
                self.img_server.save_img(user_name + "/" + str(i) + ".jpg", img)

            print("new user success ...")
            return True

    def remove_user(self, user_name):

        user_data = self.user_tb.get_user_data_by_name(user_name)

        if user_data is None:
            print("user '{}' isn't exist ...".format(user_name))
            return False

        else:
            print("remove user '{}' ...".format(user_name))

            # remove user from user table
            self.user_tb.remove_user(user_data["uid"])

            # remove face embs from emb table
            for eid in user_data["eids"]:
                self.emb_tb.remove_emb(eid)

            # remove face img from img base
            self.img_server.remove_img(user_name)

            print("remove user success ...")
            return True

    def get_user_by_uid(self, uid):
        return self.user_tb.get_user_data_by_uid(uid)

    def get_user_by_name(self, user_name):
        return self.user_tb.get_user_data_by_name(user_name)

    def get_user_imgs(self, user_name):
        return self.img_server.load_imgs(user_name)

    def get_users(self):
        return self.user_tb.get_users()

    def log_user(self, user_name, log_time):
        user = self.get_user_by_name(user_name)
        log_data = {"uid": user['uid'], "name": user_name, "log_time": log_time}
        self.log_tb.insert_user_log(log_data)

    def get_log_users(self):
        return self.log_tb.get_user_logs()

    def show_user(self):
        self.user_tb.show_table()

    def check_user_name_is_exist(self, user_name):
        user_data = self.user_tb.get_user_data_by_name(user_name)
        if user_data is None:
            return False
        else:
            return True
