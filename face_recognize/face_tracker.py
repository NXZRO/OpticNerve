import cv2

class FaceTracker:
    def __init__(self, inp_frame, inp_face_ids, inp_face_locations):
        self.face_trackers = []
        self.face_ids = inp_face_ids
        self.face_locations = inp_face_locations
        self.inp_frame = inp_frame

        self.__initial_trackers()

    def track(self, inp_frame, inp_face_locations):
        self.inp_frame = inp_frame
        self.face_locations = inp_face_locations
        tracker_ok = True

        for i, face_tracker in enumerate(self.face_trackers):

            ok, new_face_loc = face_tracker.update(self.inp_frame)
            if ok:
                self.face_locations[i] = tuple(int(point) for point in new_face_loc)  # transform to (x,y,w,h)
            else:
                tracker_ok = False

        return tracker_ok, self.face_ids, self.face_locations

    def __initial_trackers(self):
        for face_id, face_loc in zip(self.face_ids, self.face_locations):
            t = cv2.TrackerMedianFlow_create()
            t.init(self.inp_frame, face_loc)
            self.face_trackers.append(t)
