import cv2
from software import main

if __name__ == '__main__':
    cv2.namedWindow("camera", 1)
    camera = cv2.VideoCapture(0)

    while True:
        success, img = camera.read()
        cv2.imshow("camera", img)

        key = cv2.waitKey(10)
        if key == 27: #esc
            print("esc break")
            break

        if key == 32:
            file = "frames.jpg"
            cv2.imwrite(file, img)
            s = main.Get_str_from_img(file)
            print(s)

    camera.release()
    cv2.destroyWindow("camera")