from flask import Flask, g, request, render_template, make_response, Response
import cv2

from recognize_server.recognize import RecognizeServer

app = Flask(__name__)

recognize_server = RecognizeServer()

cap = cv2.VideoCapture(0)

user = ["jack"]

@app.route('/')
def index():
    return render_template('recognize.html')


def gen():
    while True:
        ok, frame = cap.read()

        # frame = recognize_server.recognize(frame)

        frame = cv2.resize(frame, (1280, 720))

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
