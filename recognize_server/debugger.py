import cv2
from recognize_server.recognize import RecognizeServer

if __name__ == '__main__':
    server = RecognizeServer()

    camera = cv2.VideoCapture(0)  # 0 -> first camera

    while True:

        ret, frame = camera.read()  # get frame

        frame = server.recognize(frame)

        cv2.imshow('frame', frame)  # show frame in window

        # press 'q' to stop
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    camera.release()  # camera release
    cv2.destroyAllWindows()  # close windows
