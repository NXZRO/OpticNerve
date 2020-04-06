from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
import base64

from user_service.user_server import UserServer
from user_service.user import User

app = Flask(__name__)


camera = cv2.VideoCapture(0)  # 0 -> first camera

user_server = UserServer()

user = User()

tmp_user = User()


@app.route('/')
def index():
    # return render_template('/admin/admin.html')

    users = user_server.get_users()

    print(users)

    return render_template('/admin/test.html', users=users)


@app.route('/log')
def log():
    log_users = user_server.get_log_users()

    print(log_users)

    return render_template('/admin/log.html', log_users=log_users)


@app.route("/show/<uid>")
def show(uid):
    print("show_user")

    uid = int(uid)
    print(uid)

    user = user_server.get_user_by_uid(uid)

    print(user)

    user_name = user['name']

    user_imgs = user_server.get_user_imgs(user_name)

    user_img_urls = []
    for user_img in user_imgs:
        user_img_urls.append(encode_img_data(user_img))

    return render_template('/admin/user.html', user=user, user_img_urls=user_img_urls)


@app.route("/remove/<uid>")
def remove(uid):
    print("remove_user")

    uid = int(uid)
    print(uid)

    user = user_server.get_user_by_uid(uid)

    print(user)

    user_name = user['name']

    user_server.remove_user(user_name)

    users = user_server.get_users()

    return render_template('/admin/test.html', users=users)


def encode_img_data(frame):
    img_bytes = cv2.imencode('.png', frame)[1]  # encode frame to bytes

    img_str = base64.b64encode(img_bytes).decode('utf-8')  # base64 encode bytes and utf8 decode bytes to string

    img_data = "data:image/png;base64," + img_str  # transform img data

    return img_data


if __name__ == '__main__':
    app.run()
