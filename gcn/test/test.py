import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)
# from sklearn.metrics import confusion_matrix
# y_true = [[2, 0, 2], [2, 0, 1]]
# y_pred = [[0, 0, 2], [2, 0, 2]]
# a= confusion_matrix(y_true, y_pred, labels=[0,1])
# print(a)
from gcn.test import extract_matrix
import json
from collections import defaultdict
# print(json.loads(a), type(a))
features, graph, lables = extract_matrix.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
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