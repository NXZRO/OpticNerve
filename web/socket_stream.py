from flask import Flask, render_template,Response
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64

# from recognize_server.recognize import RecognizeServer

app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(app)

# recognize_server = RecognizeServer()

frame_buffer_list = []



@socketio.on('connect event')
def connect(json):
    print('received json: ' + str(json))
    socketio.emit('message', {'data': 'connect success!'})


@socketio.on('update event')
def update(json):
    print('update received json')

    socketio.emit('message', {'data': 'get frame!'})

    img_data = json['data']

    frame = decode_img_data(img_data)

    # frame = recognize_server.recognize(frame)

    frame_buffer_list.append(frame)


@app.route('/')
def index():
    return render_template('index.html')


def gen():
    while True:
        if frame_buffer_list:
            frame = frame_buffer_list.pop(0)
            img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
            img_bytes = img_array.tostring()

            yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def decode_img_data(img_data):
    img_str = img_data.replace("data:image/png;base64,", "")  # get img data string

    img_bytes = base64.b64decode(img_str)  # decode base64 string to bytes

    frame = cv2.imdecode(np.fromstring(img_bytes, np.uint8), 1)  # transform to cv2 img (np array)

    return frame


def encode_img_data(frame):
    img_bytes = cv2.imencode('.png', frame)[1]  # encode frame to bytes

    img_str = base64.b64encode(img_bytes).decode('utf-8')  # base64 encode bytes and utf8 decode bytes to string

    img_data = "data:image/png;base64," + img_str  # transform img data

    return img_data


if __name__ == '__main__':
    socketio.run(app)
