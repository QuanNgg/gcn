from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform
import sys
np.set_printoptions(threshold=sys.maxsize)
# def create_graph(size):
#     graph = defaultdict(list)
#     for i in range(0, size, 5):
#         for j in range(i, i+5):
#             if j != i:
#                 graph[i].append(j)
#             if j != i+1:
#                 graph[i+1].append(j)
#             if j != i+2:
#                 graph[i+2].append(j)
#             if j != i+3:
#                 graph[i+3].append(j)
#             if j != i+4:
#                 graph[i+4].append(j)
#
#     return graph


def create_label(all_size, so_size, hoten_size, ngaysinh_size, quequan_size, hktt_size):
    # print(size)
    matrix = np.zeros((all_size,5), dtype= int)
    for row in range(all_size):
        # for col in range(5):
        #     if row % 5 == col:
        #         matrix[row][col] = 1
        if row < so_size:
            matrix[row][0] = 1

        elif row < so_size + hoten_size:
            matrix[row][1] = 1

        elif row < so_size + hoten_size + ngaysinh_size:
            matrix[row][2] = 1

        elif row < so_size + hoten_size + ngaysinh_size + quequan_size:
            matrix[row][3] = 1

        else:
            matrix[row][4] = 1

    return matrix


def get_pos_label(k, temp_line, num_line_item):
    rs = []
    label = -1
    rs_unlabel = []
    if num_line_item == 10:
        if k == 4:
            label = 0
            so = temp_line[:1]
            rs = so
            for _i in temp_line[1:]:
                rs_unlabel.append(_i)

        elif k == 5:
            label = 1
            ho_ten = temp_line[:2]
            rs = ho_ten
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ho ten', rs)
        elif k == 6:
            label = 2
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ngay_sinh', rs)
        elif k == 7:
            label = 3
            que_quan = temp_line[:2]
            rs = que_quan
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('que_quan', rs)
        elif k == 9:
            label = 4
            dk_tt = temp_line[:4]
            rs = dk_tt
            for _i in temp_line[4:]:
                rs_unlabel.append(_i)

    elif num_line_item == 11:
        if k == 4:
            label = 0
            so = temp_line[:1]
            rs = so
            for _i in temp_line[1:]:
                rs_unlabel.append(_i)
            # print('so', rs)
        elif k == 5:
            label = 1
            ho_ten = temp_line[:2]
            rs = ho_ten
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
        elif k == 7:
            label = 2
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ngay_sinh', rs)
        elif k == 8:
            label = 3
            que_quan = temp_line[:2]
            rs = que_quan
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('que_quan', rs)
        elif k == 10:
            label = 4
            dk_tt = temp_line[:4]
            rs = dk_tt
            for _i in temp_line[4:]:
                rs_unlabel.append(_i)

    elif num_line_item == 9:
        if k == 4:
            label = 0
            so = temp_line[:1]
            rs = so
            for _i in temp_line[1:]:
                rs_unlabel.append(_i)
            # print('so', rs)
        elif k == 5:
            label = 1
            ho_ten = temp_line[:2]
            rs = ho_ten
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ho ten', rs)
        elif k == 6:
            label = 2
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ngay_sinh', rs)
        elif k == 7:
            label = 3
            que_quan = temp_line[:2]
            rs = que_quan
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('que_quan', rs)
        elif k == 8:
            label = 4
            dk_tt = temp_line[:4]
            rs = dk_tt
            for _i in temp_line[4:]:
                rs_unlabel.append(_i)

    if len(rs) == 0:
        return False, 0

    pos = get_pos(rs)
    if label == 0 and pos[1] > 220:
        return False, 0
    elif label == 1 and pos[1] < 220:
        return False, 0
    elif label == 2 and (pos[1] < 355 or pos[1] > 410):
        return False, 0
    elif label == 3 and (pos[1] < 410 or pos[1] > 490):
        return False, 0

    return pos, label

def get_pos(temp_line):
    arr_num = []
    if len(temp_line) < 2:
        arr_num1 = temp_line[0].split(" ")
        for arr in arr_num1:
            arr_num.append(int(arr))
    else:
        first_box = []
        last_box = []
        first_box_t = temp_line[0].split()
        last_box_t = temp_line[-1].split()
        for arr in first_box_t:
            first_box.append(int(arr))
        for arr in last_box_t:
            last_box.append(int(arr))

        arr_num = [first_box[0], first_box[1], last_box[2], last_box[3], last_box[4], last_box[5], first_box[6], first_box[7]]

    min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
    min_x = min(min_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
    min_y = min(min_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])
    max_x = max(max_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
    max_y = max(max_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])

    pos_left_down = [min_x , max_y]
    pos_right_top = [max_x, min_y]
    pos_x = pos_left_down[0] + (pos_right_top[0] - pos_left_down[0])/2
    pos_y = pos_right_top[1] + (pos_left_down[1] - pos_right_top[1])/2

    return [pos_x, pos_y]


def get_data_from_file(file_text, file_pos):

    with open(file_text, encoding="utf8") as f1, open(file_pos, encoding="utf8") as f2:
        lines = f1.readlines()
        pos_lines = f2.readlines()
        last = -1

        arr_so = []
        arr_ngaysinh = []
        arr_hoten = []
        arr_quequan = []
        arr_hktt = []

        pos = []
        for num_line, line in enumerate(lines):
            if num_line <= last:
                continue
            line_of_one_cmnd = int(line.split("\t")[1])
            # item_name = line.split("\t")[0]
            k = 0
            while(k <= line_of_one_cmnd):
                # read file position
                num = num_line + k
                temp_line = pos_lines[num].split(";")
                # print('temp_line', temp_line)
                t_pos, label = get_pos_label(k, temp_line, line_of_one_cmnd)

                k += 1

                if t_pos == False:
                    continue
                if label == -1:
                    continue
                elif label == 0:
                    arr_so.append(t_pos)
                elif label == 1:
                    arr_hoten.append(t_pos)
                elif label == 2:
                    arr_ngaysinh.append(t_pos)
                elif label == 3:
                    arr_quequan.append(t_pos)
                elif label == 4:
                    arr_hktt.append(t_pos)

            last = num_line + int(line_of_one_cmnd)
            if num_line > 12000:
                break
            # break

        pos = arr_so + arr_hoten + arr_ngaysinh + arr_quequan + arr_hktt
        matrix_label = create_label(len(pos), len(arr_so), len(arr_hoten), len(arr_ngaysinh), len(arr_quequan), len(arr_hktt))

        feature = sparse.csr_matrix(pos)
        adj = feature.toarray()
        B = squareform(pdist(adj))
        adj = sparse.csr_matrix(B)

        idx_train = len(pos)

    return feature, matrix_label, adj, idx_train
    # return arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt


# get_data_from_file('../raw/text_1.txt', '../raw/pos_1.txt')
