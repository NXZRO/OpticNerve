import cv2


class FaceTracker:
    def __init__(self, inp_frame, inp_face_dict):
        self.face_trackers = []
        self.face_dict = inp_face_dict
        self.face_tracker_dict = {}
        self.inp_frame = inp_frame

        self.__initial_trackers()

    def track(self, inp_frame):
        self.inp_frame = inp_frame
        tracker_ok = 1

        for face_tracker, ID in zip(self.face_trackers, self.face_dict.keys()):
            ok, face_location = face_tracker.update(self.inp_frame)
            if ok:
                face_location = [int(face_loc) for face_loc in face_location]  # transform type like [x, y, w, h]
                self.face_tracker_dict[ID] = face_location
            else:
                tracker_ok = 0

        return tracker_ok, self.face_tracker_dict

    def __initial_trackers(self):
        for face_loc in self.face_dict.values():
            t = cv2.TrackerMedianFlow_create()
            t.init(self.inp_frame, tuple(face_loc))
            self.face_trackers.append(t)

