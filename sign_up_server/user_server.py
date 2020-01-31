from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector
from database_server.db_io import DataBaseIO
from sign_up_server.user import User
import cv2


class UserServer:

    def __init__(self):
        self.database_io = DataBaseIO()
        self.user = None

        self.user_name_table = {}
        self.user_table = {}

    def new_user(self, user_name, user_face_embs):
        self.user = User()
        self.user.name = user_name
        self.user.face_embs = user_face_embs
        self.user.info = {"name": self.user.name, "face_embs": self.user.face_embs}

        self.__new_uid()
        self.__new_user_info()

    def __new_uid(self):
        # search user name table by user name, and new/get uid
        self.user_name_table = self.database_io.load_user_name_table()
        self.user.uid = self.user_name_table.setdefault(self.user.name, len(self.user_name_table))
        self.database_io.save_user_name_table(self.user_name_table)

    def __new_user_info(self):
        # search user table by uid, and new/update user info
        self.user_table = self.database_io.load_user_table()
        self.user_table.setdefault(self.user.uid, self.user.info)
        self.database_io.save_user_table(self.user_table)


def draw_face(frame, face_locations):
    color = (255, 0, 0)
    for face_loc in face_locations:
        (x, y, w, h) = face_loc
        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame


if __name__ == '__main__':

    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database_io = DataBaseIO()
    user_server = UserServer()

    # # read data and new users
    # user_names = []
    # user_names = database_io.read_ID(user_names)
    #
    # img_files = database_io.get_img_base_files_path()
    #
    # for user_name, img_file in zip(user_names, img_files):
    #     img = database_io.load_img(img_file)
    #     face_locs = detector.detect(img)
    #     user_face_embs = recognizer.recognize(img, face_locs)
    #     frame = draw_face(img, face_locs)
    #     user_server.new_user(user_name, user_face_embs)
    #     database_io.save_check_img(user_name + ".jpg", frame)
    #
    # # show table
    # user_table = database_io.load_user_table()
    # print(user_table)
    #
    # user_name_table = database_io.load_user_name_table()
    # print(user_name_table)

    # ---------------------------------------------------
    # new one user
    user_name = "Obama"
    img = database_io.load_img("0.jpg")

    # detect ,recognize and save img
    face_locs = detector.detect(img)
    user_face_embs = recognizer.recognize(img, face_locs)
    frame = draw_face(img, face_locs)
    database_io.save_check_img(user_name + ".jpg", frame)

    # new user
    user_server.new_user("Obama", user_face_embs)

    # show table
    user_table = database_io.load_user_table()
    print(user_table)

    user_name_table = database_io.load_user_name_table()
    print(user_name_table)






