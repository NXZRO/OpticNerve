import socket
import numpy as np
import cv2


def decode_img_data(img_bytes):
    frame = cv2.imdecode(np.fromstring(img_bytes, np.uint8), 1)  # transform to cv2 img (np array)

    return frame


def encode_img_data(frame):
    img_bytes = cv2.imencode('.jpg', frame)[1]  # encode frame to bytes

    return img_bytes


if __name__ == '__main__':

    camera = cv2.VideoCapture(0)

    bind_ip = "127.0.0.1"
    bind_port = 9090

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    server.bind((bind_ip, bind_port))

    server.listen(5)

    print("[*] Listening on %s:%d " % (bind_ip, bind_port))

    while True:
        client, addr = server.accept()

        print('Connected by ', addr)

        while True:
            ok, frame = camera.read()

            if ok:
                img_bytes = encode_img_data(frame)

                print('img size:', len(img_bytes))

                client.send(img_bytes)

                data = client.recv(1024)
                print("client send : %s " % (data))
