from database_server.db_server import DataBaseServer

DATA_BASE_PATH = "../data_base/"
IMG_BASE_PATH = DATA_BASE_PATH + "img_base/"
CHECK_IMG_BASE_PATH = DATA_BASE_PATH + "check_img_base/"

ID_FILE = DATA_BASE_PATH + "ID.txt"
USER_TABLE = DATA_BASE_PATH + "user_table.pk"
USER_NAME_TABLE = DATA_BASE_PATH + "user_name_table.pk"
EMB_TABLE = DATA_BASE_PATH + "emb_table.pk"

INDEX_FILE = DATA_BASE_PATH + "index"
PARAMS_FILE = DATA_BASE_PATH + "params.pk"

if __name__ == '__main__':
    server = DataBaseServer()
    user_name_table = server.io.load_table(USER_NAME_TABLE)
    print(user_name_table)
    uid = user_name_table["Obama"]
    user_table = server.io.load_table(USER_TABLE)
    user_info = user_table[uid]
    target_emb = user_info[1]

    op = 1
    if op == 0:
        server.build_database()
    else:
        server.load_database()

    user_names = server.search_database([target_emb])
    print(user_names[0])
