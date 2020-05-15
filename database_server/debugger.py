from database_server.mongo_server import *
from database_server.img_server import *
from database_server.flann_server import *


def clear_database():
    db = MongoServer()
    db.reset()

    img_server = ImgServer()
    img_server.reset()

    flann_server = FlannServer()
    flann_server.reset()


if __name__ == '__main__':
    db = MongoServer()
    clear_database()

    # initial table
    id_tb = IdTable()
    emb_tb = EmbTable()
    user_tb = UserTable()

    # show tables
    id_tb.show_table()
    user_tb.show_table()
    # emb_tb.show_table()
