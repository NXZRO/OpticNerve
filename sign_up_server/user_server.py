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

    def new_user(self, user_name, user_face_embs, user_face_imgs):
        self.user = User()
        self.user.name = user_name
        self.user.face_embs = user_face_embs
        self.user.face_imgs = user_face_imgs
        self.user.info = {"name": self.user.name, "face_embs": self.user.face_embs}

        self.__new_uid()
        self.__new_user_info()
        self.__save_user_face_img()

    def __new_uid(self):
        # search user name table by user name, and new/get uid
        self.user_name_table = self.database_io.load_user_name_table()
        self.user.uid = self.user_name_table.setdefault(self.user.name, len(self.user_name_table))
        self.database_io.save_user_name_table(self.user_name_table)

    def __new_user_info(self):
        # search user table by uid, and new user info (exist no update)
        self.user_table = self.database_io.load_user_table()
        self.user_table.setdefault(self.user.uid, self.user.info)
        self.database_io.save_user_table(self.user_table)

    def __save_user_face_img(self):
        for i, img in enumerate(self.user.face_imgs):
            self.database_io.save_img(self.user.name + "/" + str(i) + ".jpg", img)


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

    # new one user
    user_name = "Obama"
    img = cv2.imread("./test_img_base/0/0.jpg")

    # detect ,recognize and save img
    face_locs = detector.detect(img)
    user_face_embs = recognizer.recognize(img, face_locs)
    frame = draw_face(img, face_locs)

    # new user
    user_server.new_user("Obama", user_face_embs,img)

    # show table
    user_table = database_io.load_user_table()
    print(user_table)

    user_name_table = database_io.load_user_name_table()
    print(user_name_table)






