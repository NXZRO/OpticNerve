from flask import Flask, render_template, Response, request, redirect, url_for
from flask import jsonify
import base64
import cv2
from recognize_server.face_recognizer import FaceRecognizer
from management_server.user_server import UserServer, User
import time
import threading

app = Flask(__name__)

user_server = UserServer()

user = User()

tmp_user = User()

recognizer = None


@app.route('/')
def home():
    return render_template('/home/home.html')


@app.route('/recognize')
def recognize():

    global recognizer

    recognizer = FaceRecognizer()

    return render_template('/recognize/recognize.html')


@app.route('/recognize/video_feed')
def recognize_video_feed():
    return Response(recognize_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


def recognize_gen():

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    if camera.read()[0] is False:
        raise Exception("Not find camera device, please open/plugin camera (webcam) !!")

    while True:
        a = time.time()
        ok, frame = camera.read()  # get frame

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


def log_user(face_ids):
    log_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

    for id in face_ids:
        if id != '':
            user_server.log_user(id, log_time)


@app.route('/sign_up')
def sign_up():
    global user, tmp_user, recognizer

    recognizer = FaceRecognizer(recognize_flag=False)

    user = User()

    tmp_user = User()

    return render_template('/sign_up/input_name.html', alert='')


@app.route("/sign_up/info", methods=['POST'])
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
        return render_template('/sign_up/input_name.html', alert="username '" + username + "' is exist !!")


@app.route("/sign_up/get_departments", methods=['POST'])
def get_departments():
    college = request.values['college']

    departments = user_server.get_departments(college)

    return jsonify({"departments": departments})


def sign_up_gen():

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    if camera.read()[0] is False:
        raise Exception("Not find camera device, please open/plugin camera (webcam) !!")

    while True:
        ret, frame = camera.read()  # get frame

        frame = cv2.resize(frame, (480, 480))

        frame, face_embs = recognizer.embedding(frame)

        tmp_user.face_embs = face_embs

        tmp_user.face_imgs = frame

        img_array = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes
        img_bytes = img_array.tostring()

        yield (b'--frame\r\n' + b'Content-Type: image/jpeg\r\n\r\n' + img_bytes + b'\r\n')


@app.route('/sign_up/video_feed')
def sign_up_video_feed():
    return Response(sign_up_gen(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/sign_up/capture", methods=['POST'])
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


@app.route('/sign_up/capture_finish', methods=['POST'])
def capture_finish():
    if len(tmp_user.face_embs) == 0:

        return render_template('/sign_up/capture_face.html', face_number=0)

    else:
        user.face_imgs.append(tmp_user.face_imgs)
        user.face_embs.append(tmp_user.face_embs[0])

        face_number = len(user.face_imgs)
        print(face_number)

        return render_template('/sign_up/capture_finish.html', face_number=face_number)


@app.route('/sign_up/finish', methods=['POST'])
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
        if user_server.check_user_name_is_exist(username) is False:
            return True
        else:
            return False


@app.route('/management')
def management():
    users = user_server.get_users()

    return render_template('/management/mangement.html', users=users)


@app.route('/management/log')
def log():
    log_users = user_server.get_log_users()
    log_users.reverse()

    return render_template('/management/log.html', log_users=log_users)


@app.route('/management/reset_database')
def reset_database():
    user_server.reset_database()
    users = user_server.get_users()

    return render_template('/management/mangement.html', users=users)


@app.route("/management/show/<uid>")
def show(uid):
    print("show_user")

    uid = int(uid)

    print(uid)

    user = user_server.get_user_by_uid(uid)

    user_imgs = user_server.get_user_imgs(user['name'])

    eids = user['eids']

    face_embs = user_server.get_face_embs(eids)

    emb_infos = []
    for eid, emb in zip(eids, face_embs):
        emb_infos.append({'eid': eid, 'face_emb': emb})

    user_img_urls = []
    for user_img in user_imgs:
        user_img_urls.append(encode_img_data(user_img))

    return render_template('/management/user.html', user=user, emb_infos=emb_infos, user_img_urls=user_img_urls)


@app.route("/management/remove/<uid>")
def remove(uid):
    print("remove_user")

    uid = int(uid)

    print(uid)

    user = user_server.get_user_by_uid(uid)

    user_name = user['name']

    user_server.remove_user(user_name)

    users = user_server.get_users()

    return render_template('/management/mangement.html', users=users)


def encode_img_data(frame):
    img_bytes = cv2.imencode('.png', frame)[1]  # encode frame to bytes

    img_str = base64.b64encode(img_bytes).decode('utf-8')  # base64 encode bytes and utf8 decode bytes to string

    img_data = "data:image/png;base64," + img_str  # transform img data

    return img_data


if __name__ == '__main__':
    app.run()
