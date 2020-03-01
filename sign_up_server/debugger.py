from sign_up_server.face_capturer import FaceCapturer
from sign_up_server.user_server import UserServer
import cv2
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
        img_dir = "./test_img_base/" + dir + "/"
        face_cap.reset()

        for img_file in os.listdir(img_dir):
            frame = cv2.imread(img_dir + img_file)
            face_cap.capture_face(frame)

        user_face_embs = face_cap.face_embs
        user_faces_imgs = face_cap.face_imgs

        user_server.new_user(user_name, user_face_embs, user_faces_imgs)  # new user


def test_delete_user(inp_user_name):
    user_server = UserServer()
    user_server.remove_user(inp_user_name)


def test_show_user():
    user_server = UserServer()
    user_server.show_user()


if __name__ == '__main__':

    # sign up from test img base people
    test_sign_up_dataset()

    # delete user
    # user_name = "jacks"
    # test_delete_user(user_name)

    # show user
    # test_show_user()






