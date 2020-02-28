from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
from sign_up_server.face_capturer import FaceCapturer
from sign_up_server.user_server import UserServer
from sign_up_server.user import User
app = Flask(__name__)


camera = cv2.VideoCapture(0)  # 0 -> first camera

user_server = UserServer()

user = User()

face_capturer = FaceCapturer()


def gen():

    while True:
        ret, frame = camera.read()  # get frame

        frame = face_capturer.detect_face(frame)

        # frame, face_ids = recognizer.recognize(frame)

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

    user.name = username

    print("username: {}".format(username))

    data = {'info': username}

    return jsonify(data)


@app.route('/capture')
def capture():
    print("capture")

    ret, frame = camera.read()   # capture frame

    ok = face_capturer.capture_face(frame)

    data = {'capture_result': ok, 'capture_num': face_capturer.capture_face_num}

    return jsonify(data)


@app.route('/finish')
def finish():
    print("finish")

    for i, img in enumerate(face_capturer.face_imgs):
        cv2.imwrite("./tmp/" + str(i) + ".jpg", img)

    print(user.name)

    user.face_embs = face_capturer.face_embs

    user.faces_imgs = face_capturer.face_imgs

    user_server.new_user(user.name, user.face_embs, user.faces_imgs)

    face_capturer.reset()

    data = {'sign_up_finish': True}

    return jsonify(data)


if __name__ == '__main__':
    app.run()
