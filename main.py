import cv2
from face_trainer import FaceTrainer
from face_detector import FaceDetector
from face_recognizer import FaceRecognizer
from face_tracker import FaceTracker

DATA_BASE_PATH = "./data_base/"

DATA_BASE_FILE = DATA_BASE_PATH + 'Data.json'


if __name__ == '__main__':

    prev_face_num = 0
    curr_face_num = 0
    tracker_flag = 0
    face_locations = []

    # initial trainer, detector, recognizer, tracker
    face_trainer = FaceTrainer()
    face_detector = FaceDetector()
    face_recognizer = FaceRecognizer()
    face_tracker = None

    face_trainer.training_data()

    face_recognizer.load_data_base(DATA_BASE_FILE)

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        face_locations = face_detector.detect(frame)  # face detect

        curr_face_num = len(face_locations)

        if curr_face_num:   # there are faces in the frame
            if tracker_flag == 0 or curr_face_num != prev_face_num:  # no tracker or no new face in the frame
                print("recognizing ...")
                frame = face_recognizer.recognize(frame, face_locations)  # face recognize

                if face_recognizer.face_dict:
                    tracker_flag = 1                                              # set next frame use tracking
                    prev_face_num = curr_face_num                                 # record face num
                    face_tracker = FaceTracker(frame, face_recognizer.face_dict)  # initial trackers

            else:
                print("tracking ...")
                ok, frame = face_tracker.track(frame)  # face track

                if not ok:
                    tracker_flag = 0   # set next frame use recognizing

        else:
            print("detecting ...")
            tracker_flag = 0  # set next frame use recognizing
            cv2.putText(frame, "Detecting...", (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                        1, (255, 0, 0), 1, cv2.LINE_AA)

        # write face num into frame
        cv2.putText(frame, "face_num : " + str(curr_face_num), (0, 60), cv2.FONT_HERSHEY_DUPLEX,
                    1, (255, 0, 0), 1, cv2.LINE_AA)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
