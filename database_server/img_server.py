import os
import cv2
import shutil

PACKAGE_PATH = os.path.dirname(__file__)

DATA_BASE_PATH = PACKAGE_PATH + "/data_base"
IMG_BASE_PATH = DATA_BASE_PATH + "/img_base/"


class ImgServer:
    def __init__(self):
        self.__Img_Base_Path = IMG_BASE_PATH

    def load_img(self, file_name):
        img = cv2.imread(self.__Img_Base_Path + file_name)
        return img

    def save_img(self, file_name, frame):
        dir = file_name.rsplit("/", 1)

        if not os.path.isdir(self.__Img_Base_Path + dir[0]):
            os.mkdir(self.__Img_Base_Path + dir[0])

        cv2.imwrite(self.__Img_Base_Path + file_name, frame)

    def remove_img(self, file_name):
        shutil.rmtree(self.__Img_Base_Path + file_name)  # delete user img dir

    def reset(self):
        # delete img base dir
        shutil.rmtree(self.__Img_Base_Path)

        # create new img dir
        os.mkdir(self.__Img_Base_Path)


if __name__ == '__main__':
    img_server = ImgServer()
    img_server.reset()

