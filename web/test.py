from flask import Flask, render_template, Response
from flask_socketio import SocketIO, emit
import cv2
import threading
import time

from recognize_server.recognize import RecognizeServer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

recognizer = RecognizeServer()

face_id_cache = []

t0 = time.time()


@socketio.on('connect event')
def connect(json):
    print('received json: ' + str(json))
    socketio.emit('log_msg', {'data': 'connect success!'})


def gen():
    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:
        ret, frame = camera.read()  # get frame

        frame, face_ids = recognizer.recognize(frame)

        td = threading.Thread(target=push_user_data_msg, args=(face_ids,))

        td.start()

        frame = cv2.resize(frame, (1280, 720))

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')

        td.join()


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def push_user_data_msg(face_ids):
    print("Child thread:")
    print(face_id_cache)
    print(face_ids)
    t = int(time.time() - t0)

    if t % 10 == 0:
        face_id_cache.clear()

    for id in face_ids:
        if id not in face_id_cache:
            face_id_cache.append(id)
            if id != '':
                socketio.emit('user_data_msg', {'data': id})


@app.route('/')
def index():
    return render_template('/recognize/test.html')


if __name__ == '__main__':
    socketio.run(app)
