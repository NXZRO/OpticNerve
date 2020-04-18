from flask import Flask, render_template, Response
import cv2

app = Flask(__name__)

cap = cv2.VideoCapture(0)

@app.route('/')
def index():
    return render_template('local_stream.html')


def gen():
    while True:
        ok, frame = cap.read()

        # draw text
        cv2.putText(frame, "Local Stream", (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/video_feed')
def video_feed():
    return Response(gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run()
