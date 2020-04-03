from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
from face_recognize.face_recognizer import FaceRecognizer
from sign_up_server.user_server import UserServer
from sign_up_server.user import User
app = Flask(__name__)


camera = cv2.VideoCapture(0)  # 0 -> first camera

user_server = UserServer()

user = User()

tmp_user = User()

recognizer = FaceRecognizer(recognize_flag=False)


def gen():

    while True:
        ret, frame = camera.read()  # get frame

        frame = cv2.resize(frame, (480, 480))

        frame, face_embs = recognizer.embedding(frame)

        tmp_user.face_embs = face_embs

        tmp_user.face_imgs = frame

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('/sign_up/sign_up.html')


@app.route("/info", methods=['POST'])
def info():
    print("info")

    username = request.values['username']

    exist = user_server.check_user_name_is_exist(username)

    if exist:
        print("username: '{}' is exist".format(username))
        data = {'resp': "/info",
                'result': False,
                'msg': "username '{}' is exist".format(username)}

    else:
        user.name = username
        print("username: {}".format(username))
        data = {'resp': "/info",
                'result': True,
                'msg': "check username '{}' is ok".format(username)}

    return jsonify(data)


@app.route('/capture')
def capture():
    print("capture")

    user.face_imgs.append(tmp_user.face_imgs)
    user.face_embs.append(tmp_user.face_embs[0])

    data = {'resp': '/capture',
            'result': True,
            'value': len(user.face_imgs)}

    return jsonify(data)


@app.route('/finish')
def finish():
    print("finish")

    ok = user_server.new_user(user.name, user.face_embs, user.face_imgs)

    user.name = ""
    user.face_imgs = []
    user.face_embs = []

    data = {'resp': '/finish',
            'result': ok}

    return jsonify(data)


if __name__ == '__main__':
    app.run()
