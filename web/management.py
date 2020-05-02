from flask import Flask, render_template, Response, request
from flask import jsonify
import cv2
import base64

from management_server.user_server import UserServer

app = Flask(__name__)

user_server = UserServer()

@app.route('/')
def index():
    users = user_server.get_users()

    return render_template('/management/mangement.html', users=users)


@app.route('/log')
def log():
    log_users = user_server.get_log_users()

    return render_template('/management/log.html', log_users=log_users)


@app.route('/reset_database')
def reset_database():
    user_server.reset_database()
    users = user_server.get_users()

    return render_template('/management/mangement.html', users=users)


@app.route("/show/<uid>")
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


@app.route("/remove/<uid>")
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
