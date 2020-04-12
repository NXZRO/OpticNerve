from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
from recognize_server.face_recognizer import FaceRecognizer
from management_server.user_server import UserServer, User

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
    return render_template('/sign_up/input_name.html')


@app.route("/info", methods=['POST'])
def info():
    username = request.values['username']

    print(username)

    ok = check_username(username)

    if ok:

        user.name = username

        colleges = user_server.get_colleges()

        departments = user_server.get_departments(colleges[0])

        return render_template('/sign_up/input_info.html', colleges=colleges, departments=departments)

    else:
        return render_template('/sign_up/input_name.html')


@app.route("/get_departments", methods=['POST'])
def get_departments():
    college = request.values['college']

    departments = user_server.get_departments(college)

    return jsonify({"departments": departments})


@app.route("/capture", methods=['POST'])
def capture():
    title = request.values['title']
    college = request.values['college']
    department = request.values['department']

    print(title)
    print(college)
    print(department)

    user.title = title
    user.college = college
    user.department = department

    return render_template('/sign_up/capture_face.html', face_number=0)


@app.route('/capture_finish', methods=['POST'])
def capture_finish():

    if len(tmp_user.face_embs) == 0:

        return render_template('/sign_up/capture_face.html', face_number=0)

    else:
        user.face_imgs.append(tmp_user.face_imgs)
        user.face_embs.append(tmp_user.face_embs[0])

        face_number = len(user.face_imgs)
        print(face_number)

        return render_template('/sign_up/capture_finish.html', face_number=face_number)


@app.route('/finish', methods=['POST'])
def finish():
    print("finish")

    ok = user_server.new_user(user)

    if ok:
        return render_template('/sign_up/finish.html')

    else:
        return "Sign up false"


def check_username(username):
    if username == '':
        return False
    else:
        return True


if __name__ == '__main__':
    app.run()
