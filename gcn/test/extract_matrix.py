from collections import defaultdict
import numpy as np
from scipy import sparse
from scipy.spatial.distance import pdist, squareform


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def create_graph(size):
    graph = defaultdict(list)
    for i in range(0, size, 5):
        for j in range(i, i+5):
            if j != i:
                graph[i].append(j)
            if j != i+1:
                graph[i+1].append(j)
            if j != i+2:
                graph[i+2].append(j)
            if j != i+3:
                graph[i+3].append(j)
            if j != i+4:
                graph[i+4].append(j)

    return graph


def create_label(size):
    # print(size)
    matrix = np.zeros((size,5), dtype= int)
    for row in range(size):
        for col in range(5):
            if row % 5 == col:
                matrix[row][col] = 1

    return matrix



def get_pos_label(k, temp_line, num_line_item):
    rs = []
    rs_unlabel = []
    if num_line_item == 10:
        if k == 4:
            so = temp_line[:1]
            rs = so
            # print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            # print('ho ten', rs)
        elif k == 6:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            # print('ngay_sinh', rs)
        elif k == 7:
            que_quan = temp_line[:2]
            rs = que_quan
            # print('que_quan', rs)
        elif k == 9:
            dk_tt = temp_line[:4]
            rs = dk_tt
            # print('dk_tt', rs)
        else:
            for i in range(len(temp_line)):
                # if i % 2 == 0:
                #     continue
                if i == 0:
                    continue
                rs_unlabel.append(temp_line[i])


    elif num_line_item == 11:
        if k == 4:
            so = temp_line[:1]
            rs = so
            for _i in temp_line[1:]:
                rs_unlabel.append(_i)
            # print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
        elif k == 7:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('ngay_sinh', rs)
        elif k == 8:
            que_quan = temp_line[:2]
            rs = que_quan
            for _i in temp_line[2:]:
                rs_unlabel.append(_i)
            # print('que_quan', rs)
        elif k == 10:
            dk_tt = temp_line[:4]
            rs = dk_tt
            for _i in temp_line[4:]:
                rs_unlabel.append(_i)
            # print('dk_tt', rs)
        else:
            for i in range(len(temp_line)):
                if i % 2 == 0:
                    continue
                # if i == 0:
                #     continue
                rs_unlabel.append(temp_line[i])

    elif num_line_item == 9:
        if k == 4:
            so = temp_line[:1]
            rs = so
            # print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            # print('ho ten', rs)
        elif k == 6:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            # print('ngay_sinh', rs)
        elif k == 7:
            que_quan = temp_line[:2]
            rs = que_quan
            # print('que_quan', rs)
        elif k == 8:
            dk_tt = temp_line[:4]
            rs = dk_tt
            # print('dk_tt', rs)
        else:
            for i in range(len(temp_line)):
                # if i % 2 == 0:
                #     continue
                if i == 0:
                    continue
                rs_unlabel.append(temp_line[i])

    arr_pos_center_label = []
    # x = []
    # y = []
    # for _rs in rs:
    #     # for i in range(len(_rs)):
    #     arr = _rs.split(" ")
    #     temp_matrix = []
    #     print(arr)
    #     for i in arr:
    #         temp_matrix.append(int(i))
    #
    #     for i in range(len(arr)):
    #         if i == 0:
    #             x.append(int(arr[i]))
    #         else:
    #             y.append(int(arr[i]))

        # if arr_num[0] > temp_matrix[0]:
        #     arr_num[0] = temp_matrix[0]
        #
        # if arr_num[1] < temp_matrix[1]:
        #     arr_num[1] = temp_matrix[1]
        #
        # if arr_num[2] > temp_matrix[2]:
        #     arr_num[2] = temp_matrix[2]
        #
        # if arr_num[3] < temp_matrix[3]:
        #     arr_num[3] = temp_matrix[3]
    if len(rs) != 0:
        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        # min_x = min(min_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
        # min_y = min(min_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])
        # max_x = max(max_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
        # max_y = max(max_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])
        # arr_num = [10000, 0, 10000, 0]
        # min_x = min(min_x, arr_num[0], arr_num[2])
        # min_y = min(min_y, arr_num[1], arr_num[3])
        # max_x = max(max_x, arr_num[0], arr_num[2])
        # max_y = max(max_y, arr_num[1], arr_num[3])

        # min_x = min(x)
        # max_x = max(x)
        # min_y = min(y)
        # max_y = max(y)

        first_arr = []
        arr_f = rs[0].split(" ")
        for _rs in arr_f:
            first_arr.append(_rs)

        arr_l = rs.pop().split(" ")
        last_arr = []
        for _rs in arr_l:
            last_arr.append(_rs)

        min_x = int(first_arr[0])
        min_y = int(first_arr[1])
        max_x = int(last_arr[4])
        max_y = int(last_arr[5])

        pos_left_down = [min_x , max_y]
        pos_right_top = [max_x, min_y]

        pos_x = pos_left_down[0] + (pos_right_top[0] - pos_left_down[0])/2
        pos_y = pos_right_top[1] + (pos_left_down[1] - pos_right_top[1])/2
        pos_center = [pos_x, pos_y]

        arr_pos_center_label.append(pos_center)

    # get pos_unlabels
    arr_pos_center_unlabel = []
    for _rs in rs_unlabel:
        # for i in range(len(_rs)):
        arr = _rs.split(" ")
        arr_num1 = []
        for a in arr:
            arr_num1.append(int(a))

        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        min_x = min(min_x, arr_num1[0], arr_num1[2], arr_num1[4], arr_num1[6])
        min_y = min(min_y, arr_num1[1], arr_num1[3], arr_num1[5], arr_num1[7])
        max_x = max(max_x, arr_num1[0], arr_num1[2], arr_num1[4], arr_num1[6])
        max_y = max(max_y, arr_num1[1], arr_num1[3], arr_num1[5], arr_num1[7])

        pos_left_down = [min_x , max_y]
        pos_right_top = [max_x, min_y]
        pos_x = pos_left_down[0] + (pos_right_top[0] - pos_left_down[0])/2
        pos_y = pos_right_top[1] + (pos_left_down[1] - pos_right_top[1])/2
        pos_center = [pos_x, pos_y]
        arr_pos_center_unlabel.append(pos_center)

        # print('pos_center', pos_center)

    return arr_pos_center_label, arr_pos_center_unlabel



def get_data_from_file(file_text, file_pos):
    first_line = True
    with open(file_text, encoding="utf8") as f1, open(file_pos, encoding="utf8") as f2:
        # first_line = f1.readline()
        # f2.write(first_line)
        lines = f1.readlines()
        pos_lines = f2.readlines()
        last = -1
        arr_pos = []
        arr_num = []
        rs_dict = {}
        pos_labels = []
        pos_unlabels = []
        pos = []
        for num_line, line in enumerate(lines):
            if num_line <= last:
                continue
            line_of_one_cmnd = int(line.split("\t")[1])
            # item_name = line.split("\t")[0]
            k = 0
            while(k <= line_of_one_cmnd):
                # read file position
                temp_line = pos_lines[k].split(";")
                arr_pos_center_label, arr_pos_center_unlabel = get_pos_label(k, temp_line, line_of_one_cmnd)
                if len(arr_pos_center_label) != 0:
                    pos_labels = pos_labels + arr_pos_center_label
                if len(arr_pos_center_unlabel) != 0:
                    pos_unlabels = pos_unlabels + arr_pos_center_unlabel

                k += 1

            last = num_line + int(line_of_one_cmnd)
            if num_line > 500:
                break
            # break

        pos = pos_labels + pos_unlabels
        graph = create_graph(len(pos))
        matrix_label = create_label(len(pos))
        feature = sparse.csr_matrix(pos)
        adj = feature.toarray()
        B = squareform(pdist(adj))
        adj = sparse.csr_matrix(B)

    return feature, graph, matrix_label, adj

# def get_pos():

# get_data_from_file('../raw/text_1.txt', '../raw/pos_1.txt')
