import os
import cv2
import shutil
import numpy as np

PACKAGE_PATH = os.path.dirname(__file__)

DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
IMG_BASE_PATH = DATA_BASE_PATH + "/img_base/"


class ImgServer:
    def __init__(self):
        self.__Img_Base_Path = IMG_BASE_PATH

    def load_img(self, file_name):
        img = cv2.imdecode(np.fromfile(self.__Img_Base_Path + file_name, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        return img

    def load_imgs(self, dir_name):
        dir = self.__Img_Base_Path + "/" + dir_name
        imgs = []
        files = os.listdir(dir)
        for file in files:
            img = cv2.imdecode(np.fromfile(dir + "/" + file, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
            imgs.append(img)
        return imgs

    def save_img(self, file_name, frame):
        dir = file_name.rsplit("/", 1)

        if not os.path.isdir(self.__Img_Base_Path + dir[0]):
            os.mkdir(self.__Img_Base_Path + dir[0])

        cv2.imencode(".jpg", frame)[1].tofile(self.__Img_Base_Path + file_name)

    def remove_img(self, file_name):
        shutil.rmtree(self.__Img_Base_Path + file_name)  # delete user img dir

    def reset(self):
        # delete img base dir
        shutil.rmtree(self.__Img_Base_Path)

        # create new img dir
        os.mkdir(self.__Img_Base_Path)


if __name__ == '__main__':
    # remove all data
    img_server = ImgServer()
    img_server.reset()
