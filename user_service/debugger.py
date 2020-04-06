from face_recognize.face_recognizer import FaceRecognizer
from user_service.user_server import UserServer
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

        # new user by user data
        user_server.new_user(user_name, user_face_embs, user_faces_imgs)


def test_sign_up_local(user_name):

    recognizer = FaceRecognizer(recognize_flag=False)

    user_server = UserServer()

    camera = cv2.VideoCapture(0)  # Fish -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        frame, face_embs = recognizer.embedding(frame)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows

    user_server.new_user(user_name, [face_embs[0]], frame)  # new user


def test_delete_user(inp_user_name):
    user_server = UserServer()
    user_server.remove_user(inp_user_name)


def test_show_user():
    user_server = UserServer()
    user_server.show_user()


if __name__ == '__main__':

    # sign up from test img base people
    test_sign_up_dataset()

    # sign up by web camera
    # user_name = "Frank"
    # test_sign_up_local(user_name)

    # delete user
    # user_name = "frank"
    # test_delete_user(user_name)

    # show user
    # test_show_user()






