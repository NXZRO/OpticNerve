from flask import Flask, render_template, Response
import cv2
import threading
import time

from recognize_server.face_recognizer import FaceRecognizer
from management_server.user_server import UserServer


app = Flask(__name__)

user_server = UserServer()

recognizer = FaceRecognizer()


def gen():

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:
        a = time.time()
        ret, frame = camera.read()  # get frame

        frame, face_ids = recognizer.recognize(frame)

        frame = cv2.resize(frame, (1280, 720))

        td = threading.Thread(target=log_user, args=(face_ids,))

        td.start()

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        td.join()

        b = time.time()
        print(b-a)


@app.route('/video_feed')
def video_feed():

    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def log_user(face_ids):
    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for id in face_ids:
        if id != '':
            user_server.log_user(id, log_time)


@app.route('/')
def index():
    return render_template('/recognize/recognize.html')


if __name__ == '__main__':
    app.run()
