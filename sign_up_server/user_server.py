from database_server.db_server import DataBase


class User:
    def __init__(self):
        self.uid = None
        self.name = ""
        self.face_embs = []
        self.face_imgs = []
        self.eids = []


class UserServer:

    def __init__(self):
        self.database = DataBase()
        self.user = None

    def new_user(self, user_name, user_face_embs, user_face_imgs):

        user_name_table = self.database.load_table("USER_NAME_TABLE")

        # check user exist
        if user_name_table.get(user_name) is not None:

            print("user '{}' is exist ...".format(user_name))

        else:
            self.user = User()
            self.user.name = user_name

            self.user.face_embs = user_face_embs
            self.user.face_imgs = user_face_imgs

            # search user name table by user name, and new uid
            uids = list(user_name_table.get_values())

            if uids == []:
                self.user.uid = 0
            else:
                self.user.uid = uids[-1] + 1

            user_name_table.new(self.user.name, self.user.uid)

            # add into emb table
            emb_table = self.database.load_table("EMB_TABLE")
            eids = list(emb_table.get_keys())

            if eids == []:
                eid = 0
            else:
                eid = eids[-1] + 1

            for emb in self.user.face_embs:
                self.user.eids.append(eid)
                emb_info = {"face_embs": emb, "uid": self.user.uid}
                emb_table.new(eid, emb_info)
                eid += 1

            # search user table by uid, and new user info
            self.user.info = {"name": self.user.name, "eids": self.user.eids, "face_embs": self.user.face_embs}
            user_table = self.database.load_table("USER_TABLE")
            user_table.new(self.user.uid, self.user.info)

            # save face img to img base
            for i, img in enumerate(self.user.face_imgs):
                self.database.save_img(self.user.name + "/" + str(i) + ".jpg", img)

            print("new user success...")

    def delete_user(self, user_name):
        user_name_table = self.database.load_table("USER_NAME_TABLE")

        uid = user_name_table.get(user_name)

        # check user exist
        if uid is None:

            print("user '{}' isn't exist ...".format(user_name))

        else:
            # remove from user name table
            user_name_table.delete(user_name)

            # remove form user table
            user_table = self.database.load_table("USER_TABLE")
            user_info = user_table.get(uid)
            eids = user_info["eids"]
            user_table.delete(uid)

            # remove from emb table
            emb_table = self.database.load_table("EMB_TABLE")
            for eid in eids:
                emb_table.delete(eid)

            # remove user img dir
            self.database.remove_img(user_name)

            print("delete user success...")
