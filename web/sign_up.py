from flask import Flask, render_template, Response
from flask import jsonify
import cv2
from sign_up_server.capturer import Capturer
app = Flask(__name__)


camera = cv2.VideoCapture(0)  # 0 -> first camera

face_capturer = Capturer()


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
    return render_template('sign_up.html')


@app.route('/capture')
def capture():
    print("capture")

    ret, frame = camera.read()   # capture frame

    ok = face_capturer.capture_face(frame)

    data = {'capture_result': ok, 'capture_num': face_capturer.capture_face_num}

    return jsonify(data)


@app.route('/capture_finish')
def capture_finish():
    print("capture_finish")

    for i, img in enumerate(face_capturer.face_imgs):
        cv2.imwrite(str(i) + ".jpg", img)

    data = {'capture_finish': True}

    return jsonify(data)


if __name__ == '__main__':
    app.run()
