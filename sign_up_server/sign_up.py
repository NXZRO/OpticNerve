from sign_up_server.face_capturer import FaceCapturer
from sign_up_server.user_server import UserServer

if __name__ == '__main__':
    user_server = UserServer()

    user_name = "qwq"

    face_cap = FaceCapturer()
    user_face_embs = face_cap.capture_face()

    user_server.new_user(user_name, user_face_embs)

    print(user_server.database_io.load_user_table())

    print(user_server.database_io.load_user_name_table())
