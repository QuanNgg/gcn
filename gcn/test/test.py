import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
# from sklearn.metrics import confusion_matrix
# y_true = [[2, 0, 2], [2, 0, 1]]
# y_pred = [[0, 0, 2], [2, 0, 2]]
# a= confusion_matrix(y_true, y_pred, labels=[0,1])
# print(a)
# from gcn.test import extract_matrix
# import json
# from collections import defaultdict
# print(json.loads(a), type(a))
# features, graph, lables = extract_matrix.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
# print(pickle.dumps(graph))

# with open('./extract_matrix/graph.txt', 'w') as f:
#     list(graph)
#     graph = json.dumps(graph)
#     f.write(graph)
#     f.close()

# with open('./extract_matrix/graph.txt') as f:
#     # lables = lables.tostring().decode('utf-8')
#     a = f.read()
#     f.close()
# b = json.loads(a)
# c = defaultdict(int,b)
# print(c)
# numpy.array(list(a), dtype=int)


import matplotlib.pyplot as plt
from gcn.test import extract_matrix_v2
# features, graph, labels, adj1, idx_train = extract_matrix.get_data_test_from_file('../raw/text_1.txt', '../raw/pos_1.txt')
# features = features.toarray()
# x = []
# y = []
# for fe in features:
#     x.append(fe[0])
#     y.append(fe[1])
# print("x", len(x), len(y))
# plt.plot(list(x),list(y), "go")q
# plt.show()
# all, arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt = extract_matrix_v2.get_pre_data('/home/hq-lg/gcn/gcn/data_cmnd/text_test.txt', '/home/hq-lg/gcn/gcn/data_cmnd/pos_test.txt')


# all, arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt = extract_matrix_v2.get_pre_data('/home/hq-lg/gcn/gcn/raw/text_1.txt', '/home/hq-lg/gcn/gcn/raw/pos_1.txt')
all, arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt = extract_matrix_v2.get_pre_data_test('/home/hq-lg/gcn/gcn/raw/text_1.txt', '/home/hq-lg/gcn/gcn/raw/pos_1.txt')

# arr_mother = [arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt]
x_so = []
y_so = []
x_hoten = []
y_hoten = []
x_ngaysinh = []
y_ngaysinh = []
x_quequan = []
y_quequan = []
x_hktt = []
y_hktt = []
for arr in arr_so:
    x_so.append(arr[0])
    y_so.append(arr[1])
for arr in arr_hoten:
    x_hoten.append(arr[0])
    y_hoten.append(arr[1])
for arr in arr_ngaysinh:
    x_ngaysinh.append(arr[0])
    y_ngaysinh.append(arr[1])
for arr in arr_quequan:
    x_quequan.append(arr[0])
    y_quequan.append(arr[1])
for arr in arr_hktt:
    x_hktt.append(arr[0])
    y_hktt.append(arr[1])
plt.xlabel('x')
plt.ylabel('y')
plt.plot(x_so, y_so, 'go', x_hoten, y_hoten, 'b*', x_ngaysinh, y_ngaysinh, 'rD', x_quequan, y_quequan, 'y^', x_hktt, y_hktt, 'kX')
plt.show()
# plt.savefig('/home/hq-lg/gcn/gcn/pictures/test.jpg')
