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

        for i, line in enumerate(lines):
            if i <= last:
                continue
            num_line = line.split("\t")[1]
            file_name = line.split("\t")[0]
            # f3.write(line.split("\t")[0])
            width = pos_lines[i].split("\t")[1]
            height = pos_lines[i].split("\t")[2]
            # print(line, num_line)
            data = []
            arr_node = []
            arr_adj = []
            arr_line = []
            for j in range(i + 1, i + int(num_line) + 1):
                temp_str = lines[j].split("\t")
                temp_str[-1] = temp_str[-1][0:-1]
                str_pre = ""
                for s in temp_str:
                    if str_pre == "":
                        str_pre += s
                    else:
                        str_pre += " " + s
                data.append(str_pre)
                last = j
                ######## read file pos
                temp_line = pos_lines[j].split(";")

                tl_x, tl_y, br_x, br_y = 10000, 10000, 0, 0
                for k in range(len(temp_line)):
                    a = temp_line[k].split(" ")
                    b = []
                    for h in range(len(a)):
                        b.append(int(a[h]))
                    tl_x = min(tl_x, b[0], b[2], b[4], b[6])
                    tl_y = min(tl_y, b[1], b[3], b[5], b[7])
                    br_x = max(br_x, b[0], b[2], b[4], b[6])
                    br_y = max(br_y, b[1], b[3], b[5], b[7])
                    x_input = (br_x - tl_x) / 2
                    y_input = (br_y - tl_y) / 2
                arr_line.append([tl_x, tl_y, br_x, br_y])
                arr_node.append([x_input,y_input])

            return (arr_node)
