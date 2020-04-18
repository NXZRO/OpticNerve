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

    HOST = '127.0.0.1'
    PORT = 9090

    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((HOST, PORT))

    while True:

        img_bytes = s.recv(100000)

        frame = decode_img_data(img_bytes)
        print('frame', frame)

        cv2.imshow('frame', frame)

        print("client recv data")

        s.send(b"ACK!")

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()  # close windows


