import cv2
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector
from database_server.db_io import DataBaseIO
from database_server.db_server import DataBaseServer


def draw_face(frame, face_locations):
    color = (255, 0, 0)
    for face_loc in face_locations:
        (x, y, w, h) = face_loc
        # draw face box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    return frame


if __name__ == '__main__':
    detector = FaceDetector()
    recognizer = FaceRecognizer()
    database_io = DataBaseIO()

    # new one user
    img = cv2.imread('0.jpg')

    # detect ,recognize and save img
    face_locs = detector.detect(img)
    user_face_embs = recognizer.recognize(img, face_locs)
    print(user_face_embs)
    frame = draw_face(img, face_locs)
    cv2.imwrite('1.jpg', frame)

    dbs = DataBaseServer()
    dbs.build_database()
    dbs.load_database()

    id = dbs.search_database(user_face_embs)

    print(id)
