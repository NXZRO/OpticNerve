import cv2


class FaceTracker:
    def __init__(self, inp_frame, inp_face_dict):
        self.face_trackers = []
        self.face_dict = inp_face_dict
        self.inp_frame = inp_frame

        self.__initial_trackers()

    def track(self, inp_frame):
        self.inp_frame = inp_frame
        tracker_ok = 1

        for face_tracker, ID in zip(self.face_trackers, self.face_dict.keys()):
            ok, face_location = face_tracker.update(self.inp_frame)
            if ok:
                self.face_dict[ID] = face_location
                (x, y, w, h) = (int(face_loc) for face_loc in face_location)
                cv2.rectangle(self.inp_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(self.inp_frame, ID, (x, y - 20), cv2.FONT_HERSHEY_DUPLEX,
                            1, (0, 0, 255), 1, cv2.LINE_AA)
            else:
                tracker_ok = 0

        cv2.putText(self.inp_frame, "Tracking...", (0, 30), cv2.FONT_HERSHEY_DUPLEX,
                    1, (0, 0, 255), 1, cv2.LINE_AA)

        return tracker_ok, self.inp_frame

    def __initial_trackers(self):
        for face_loc in self.face_dict.values():
            t = cv2.TrackerMedianFlow_create()
            t.init(self.inp_frame, tuple(face_loc))
            self.face_trackers.append(t)

