def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index

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
        for num_line, line in enumerate(lines):
            if num_line <= last:
                continue
            # i = num_line
            matrix = []
            num_line_item = int(line.split("\t")[1])
            # item_name = line.split("\t")[0]
            if num_line_item == 10 or num_line_item == 11 or num_line_item == 9:
                arr_num.append(num_line_item)
            print('num_line_item', num_line_item)
            k = 0
            while(k <= num_line_item):
                # read file position
                print('k', k)
                temp_line = pos_lines[k].split(";")
                pos_center = get_pos_label(k, temp_line, num_line_item)
                print('pos', pos_center)
                k += 1
                if pos_center is None:
                    continue
                arr_pos.append(pos_center)

            last = num_line + int(num_line_item)
            # break
        print('arr_pos', arr_pos, len(arr_pos), len(arr_num))
    return arr_pos


def get_pos_label(k, temp_line, num_line_item):
    rs = []
    if num_line_item == 10:
        if k == 4:
            so = temp_line[:1]
            rs = so
            print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            print('ho ten', rs)
        elif k == 6:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            print('ngay_sinh', rs)
        elif k == 7:
            que_quan = temp_line[:2]
            rs = que_quan
            print('que_quan', rs)
        elif k == 9:
            dk_tt = temp_line[:4]
            rs = dk_tt
            print('dk_tt', rs)

    elif num_line_item == 11:
        if k == 4:
            so = temp_line[:1]
            rs = so
            print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            print('ho ten', rs)
        elif k == 7:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            print('ngay_sinh', rs)
        elif k == 8:
            que_quan = temp_line[:2]
            rs = que_quan
            print('que_quan', rs)
        elif k == 10:
            dk_tt = temp_line[:4]
            rs = dk_tt
            print('dk_tt', rs)

    elif num_line_item == 9:
        if k == 4:
            so = temp_line[:1]
            rs = so
            print('so', rs)
        elif k == 5:
            ho_ten = temp_line[:2]
            rs = ho_ten
            print('ho ten', rs)
        elif k == 6:
            ngay_sinh = temp_line[:2]
            rs = ngay_sinh
            print('ngay_sinh', rs)
        elif k == 7:
            que_quan = temp_line[:2]
            rs = que_quan
            print('que_quan', rs)
        elif k == 8:
            dk_tt = temp_line[:4]
            rs = dk_tt
            print('dk_tt', rs)

    for i in range(len(rs)):
        arr = rs[i].split(" ")
        arr_num = []

        for j in range(len(arr)):
            arr_num.append(int(arr[j]))

        min_x, min_y, max_x, max_y = 10000, 10000, 0, 0
        min_x = min(min_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
        min_y = min(min_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])
        max_x = max(max_x, arr_num[0], arr_num[2], arr_num[4], arr_num[6])
        max_y = max(max_y, arr_num[1], arr_num[3], arr_num[5], arr_num[7])

        pos_left_down = [min_x , max_y]
        pos_right_top = [max_x, min_y]
        pos_x = pos_left_down[0] + (pos_right_top[0] - pos_left_down[0])/2
        pos_y = pos_right_top[1] + (pos_left_down[1] - pos_right_top[1])/2
        pos_center = [pos_x, pos_y]

        print('pos_center', pos_center)

        return pos_center

get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')