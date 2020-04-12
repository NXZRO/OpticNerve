import cv2
from recognize_server.face_detector import FaceDetector
from recognize_server.face_embedder import FaceRecognizer
import time

DetectingState = 0
RecognizingState = 1
TrackingState = 2


def draw_face_info(frame, face_IDs, face_locations, process_state, curr_face_num):
    color_dict = {0: (255, 0, 0), 1: (0, 255, 0), 2: (0, 0, 255)}
    color = color_dict[process_state]

    state_dict = {0: "Detecting...", 1: "Recognizing...", 2: "Tracking"}
    state = state_dict[process_state]

    # draw process state
    cv2.putText(frame, state, (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                1, color, 1, cv2.LINE_AA)

    # draw current face num
    cv2.putText(frame, "face_num : " + str(curr_face_num), (0, 60), cv2.FONT_HERSHEY_DUPLEX,
                1, color, 1, cv2.LINE_AA)

    if face_IDs:
        for (face_ID, face_loc) in zip(face_IDs, face_locations):
            (x, y, w, h) = face_loc
            # draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # draw face id
            cv2.putText(frame, str(face_ID), (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                        1, color, 1, cv2.LINE_AA)

    else:
        for face_loc in face_locations:
            (x, y, w, h) = face_loc
            # draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)


if __name__ == '__main__':

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(0)  # Fish -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        a = time.time()
        face_locations = face_detector.detect(frame)  # face detect
        b = time.time()
        print("detect: {} sec".format(b - a))

        curr_face_num = len(face_locations)

        face_IDs = ["Detect" + str(i) for i in range(curr_face_num)]
        print(face_IDs)

        if curr_face_num:  # there are faces in the frame
            print("recognizing no search databse...")
            a = time.time()
            face_embs = face_recognizer.recognize(frame, face_locations)  # face recognize
            b = time.time()
            print("recog: {} sec".format(b - a))

            draw_face_info(frame, face_IDs, face_locations, RecognizingState, curr_face_num)  # draw frame

        else:
            print("detecting ...")
            draw_face_info(frame, face_IDs, face_locations, DetectingState, curr_face_num)  # draw frame

        cv2.imshow('frame', frame)  # show frame in window

        cv2.imwrite('test2.jpg', frame)

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
    FaceDetector()
