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
        a = time.time()
        ret, frame = camera.read()  # get frame

        frame, face_ids = recognizer.recognize(frame)

        frame = cv2.resize(frame, (1280, 720))

        td = threading.Thread(target=push_user_data_msg, args=(face_ids,))

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


def push_user_data_msg(face_ids):
    print("Child thread:")
    print(face_id_cache)
    print(face_ids)

    for id in face_ids:
        if id not in face_id_cache:
            if id != '':
                face_id_cache.append(id)
                socketio.emit('append_user_card', {'data': id})

    for idx, cache_id in enumerate(face_id_cache):
        if cache_id not in face_ids:
            face_id_cache.pop(idx)
            socketio.emit('remove_user_card', {'data': cache_id})


@app.route('/')
def index():
    return render_template('/recognize/recognize.html')


if __name__ == '__main__':
    socketio.run(app)
