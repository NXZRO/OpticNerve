import cv2
from recognize_server.recognize import RecognizeServer
import time

if __name__ == '__main__':

    max_cost = 0

    first_max_cost = 0

    f = 0

    server = RecognizeServer()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        t1 = time.clock()

        frame = server.recognize(frame)  # recognize frame

        t2 = time.clock()

        cost = t2-t1
        if max_cost < cost:
            if f == 0 and cost > 1:
                f = 1
                first_max_cost = cost
            else:
                max_cost = cost

        print(cost)
        print("first max cost: {} sec".format(first_max_cost))
        print("max cost: {} sec".format(max_cost))

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
