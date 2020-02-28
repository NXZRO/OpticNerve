from sign_up_server.face_capturer import FaceCapturer
from sign_up_server.user_server import UserServer

import os


def test_sign_up_dataset():

    user_server = UserServer()
    face_cap = FaceCapturer()

    dirs = os.listdir("./test_img_base")
    user_names = []

    with open("./test_img_base/ID.txt", "r") as fp:
        for id in fp:
            user_names.append(str(id).rstrip('\n'))

    for i, (user_name, dir) in enumerate(zip(user_names, dirs)):
        print(dir)

        user_face_embs = face_cap.capture_test_imgs("./test_img_base/" + dir + "/")

        user_faces_imgs = face_cap.face_imgs

        user_server.new_user(user_name, user_face_embs, user_faces_imgs)  # new user


def test_sign_up_user(inp_user_name):
    user_server = UserServer()

    user_name = inp_user_name

    face_cap = FaceCapturer()
    user_face_embs = face_cap.capture_face()

    user_faces_imgs = face_cap.face_imgs

    user_server.new_user(user_name, user_face_embs, user_faces_imgs)


def test_delete_user(inp_user_name):
    user_server = UserServer()

    user_server.remove_user(inp_user_name)


if __name__ == '__main__':

    # sign up from test img base people
    # test_sign_up_dataset()

    # sign up yourself
    # user_name = "HUANG TING HOU"
    # test_sign_up_user(user_name)

    # delete user
    user_name = "HUANG TING HOU"
    test_delete_user(user_name)

    # delete database_server
    # user_server = UserServer()
    # user_server.database_server.clear()




