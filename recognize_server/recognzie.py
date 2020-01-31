import cv2
from face_recognize.face_recognizer import FaceRecognizer
from face_recognize.face_detector import FaceDetector
from database_server.db_server import DataBaseServer


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
    db_server = DataBaseServer()
    db_server.build_database()
    db_server.load_database()

    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        face_locations = face_detector.detect(frame)  # face detect

        curr_face_num = len(face_locations)

        face_IDs = ["Detect" + str(i) for i in range(curr_face_num)]
        print(face_IDs)

        if curr_face_num:  # there are faces in the frame
            print("recognizing no search databse...")
            face_embs = face_recognizer.recognize(frame, face_locations)                      # face recognize
            face_IDs = db_server.search_database(face_embs)                                   # search data base
            draw_face_info(frame, face_IDs, face_locations, RecognizingState, curr_face_num)  # draw frame

        else:
            print("detecting ...")
            tracker_flag = 0  # set next frame use recognizing
            draw_face_info(frame, face_IDs, face_locations, DetectingState, curr_face_num)  # draw frame

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
    FaceDetector()
