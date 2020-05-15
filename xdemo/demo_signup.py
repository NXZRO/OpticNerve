import cv2
from recognize_server.face_recognizer import FaceRecognizer
from management_server.user_server import UserServer, User
import os
import numpy as np

user_server = UserServer()

recognizer = FaceRecognizer(recognize_flag=False)


def cv2_imread(filePath):
    cv_img = cv2.imdecode(np.fromfile(filePath, dtype=np.uint8), -1)

    return cv_img


def sign_up():
    src_dir = 'xdemo_src_img'
    dirs = os.listdir(src_dir)

    for name in dirs:
        user = User()

        user.name = name
        user.title = "學生"
        user.college = "工學院"
        user.department = "資工系（日）"

        img_dir = src_dir + '/' + name
        for img_name in os.listdir(img_dir):
            img = cv2_imread(img_dir + '/' + img_name)
            frame, face_embs = recognizer.embedding(img)
            user.face_imgs.append(frame)
            user.face_embs.append(face_embs[0])

        ok = user_server.new_user(user)

        if ok:
            print("'{}' sign up success !!".format(user.name))

        else:
            print("'{}' is exist !!".format(user.name))


if __name__ == '__main__':
    sign_up()
