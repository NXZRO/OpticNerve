from recognize_server.face_recognizer import FaceRecognizer
from management_server.user_server import UserServer, User
import cv2
import os


def test_sign_up_dataset():

    recognizer = FaceRecognizer(recognize_flag=False)

    user_server = UserServer()

    dirs = os.listdir("./test_img")

    for user_name in dirs:
        img_dir = "./test_img/" + user_name

        user_face_embs = []
        user_faces_imgs = []

        for img_file in os.listdir(img_dir):
            img_path = img_dir + "/" + img_file
            print(img_path)

            # get face locations and face embeddings
            frame = cv2.imread(img_path)
            frame, face_embs = recognizer.embedding(frame)

            # store user data
            user_face_embs.append(face_embs[0])
            user_faces_imgs.append(frame)

        user = User()

        user.name = user_name

        user.face_embs = user_face_embs

        user.face_imgs = user_faces_imgs

        user.title = "TestTitle"

        user.college = "TestCollege"

        user.department = "TestDept"

        # new user by user data
        user_server.new_user(user)


def test_delete_user(inp_user_name):
    user_server = UserServer()
    user_server.remove_user(inp_user_name)


def test_show_user():
    user_server = UserServer()
    print(user_server.get_users())


if __name__ == '__main__':

    # sign up from test img base people
    test_sign_up_dataset()

    # delete user
    # user_name = "frank"
    # test_delete_user(user_name)

    # show user
    # test_show_user()






