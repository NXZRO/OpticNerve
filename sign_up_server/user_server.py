from database_server.db_io import DataBaseIO
from sign_up_server.user import User


class UserServer:

    def __init__(self):
        self.database_io = DataBaseIO()
        self.user = None

        self.user_name_table = {}
        self.user_table = {}
        self.emb_table = {}

    def new_user(self, user_name, user_face_embs, user_face_imgs):

        self.user_name_table = self.database_io.load_user_name_table()

        # check user exist
        if user_name in self.user_name_table:

            print("user '{}' is exist ...".format(user_name))

        else:

            self.user = User()
            self.user.name = user_name

            self.user.face_embs = user_face_embs
            self.user.face_imgs = user_face_imgs
            self.user.info = {"name": self.user.name, "face_embs": self.user.face_embs}

            self.__new_uid()
            self.__new_user_info()
            self.__add_emb_table()
            self.__save_user_face_img()
            print("new user success...")

    def __new_uid(self):
        # search user name table by user name, and new uid
        self.user.uid = self.user_name_table.setdefault(self.user.name, len(self.user_name_table))
        self.database_io.save_user_name_table(self.user_name_table)

    def __new_user_info(self):
        # search user table by uid, and new user info
        self.user_table = self.database_io.load_user_table()
        self.user_table.setdefault(self.user.uid, self.user.info)
        self.database_io.save_user_table(self.user_table)

    def __add_emb_table(self):
        self.emb_table = self.database_io.load_emb_table()
        for emb in self.user.face_embs:
            eid = len(self.emb_table)
            emb_info = {"face_embs": emb, "uid": self.user.uid}
            self.emb_table.setdefault(eid, emb_info)
        self.database_io.save_emb_table(self.emb_table)

    def __save_user_face_img(self):
        for i, img in enumerate(self.user.face_imgs):
            self.database_io.save_img(self.user.name + "/" + str(i) + ".jpg", img)



