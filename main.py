import cv2
from face_trainer import FaceTrainer
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker

DATA_BASE_PATH = "./data_base/"

DATA_SET_FILE = DATA_BASE_PATH + 'Data.json'

DetectingState = 0
RecognizingState = 1
TrackingState = 2


def draw_face_info(frame, face_dict, process_state, curr_face_num):
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

    if face_dict:
        for (face_ID, face_loc) in face_dict.items():
            (x, y, w, h) = face_loc
            # draw face box
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)

            # draw face id
            cv2.putText(frame, str(face_ID), (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                        1, color, 1, cv2.LINE_AA)


if __name__ == '__main__':

    prev_face_num = 0
    curr_face_num = 0
    tracker_flag = 0
    face_locations = []
    face_dict = {}

    # initial trainer, detector, recognizer, tracker
    face_trainer = FaceTrainer()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    face_tracker = None

    face_trainer.training_data()

    face_recognizer.load_data(face_trainer.data_base_dict)

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        face_locations = face_detector.detect(frame)  # face detect

        face_dict = {}

        curr_face_num = len(face_locations)

        if curr_face_num:   # there are faces in the frame
            if tracker_flag == 0 or curr_face_num != prev_face_num:  # no tracker or no new face in the frame
                print("recognizing ...")
                face_dict = face_recognizer.recognize(frame, face_locations)  # face recognize

                if face_dict:
                    tracker_flag = 1                              # set next frame use tracking
                    prev_face_num = curr_face_num                 # record face num
                    face_tracker = FaceTracker(frame, face_dict)  # initial trackers

                draw_face_info(frame, face_dict, RecognizingState, curr_face_num)  # draw frame

            else:
                print("tracking ...")
                ok, face_dict = face_tracker.track(frame)  # face track

                if not ok:
                    tracker_flag = 0  # set next frame use recognizing

                draw_face_info(frame, face_dict, TrackingState, curr_face_num)  # draw frame

        else:
            print("detecting ...")
            tracker_flag = 0  # set next frame use recognizing
            draw_face_info(frame, face_dict, DetectingState, curr_face_num)  # draw frame

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
