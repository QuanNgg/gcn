from gcn import input
# import numpy as np
from scipy import sparse
a, b, c = input.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
# print(a)
# aa = np.zeros((11))
# # print(aa)
# b = np.array(a, dtype=float)
# # print(b)
# fea = sparse.csr_matrix(b)
# print('features', fea)
# adj = []
# for i in range(11):
#     print(b[i])
#     k = []
#     for j in range(11):
#         a = np.linalg.norm(b[i] - b[j])
#         adj.append(a)
# adj = np.array(adj, dtype=int)
# adj = adj.reshape(11,11)
# print(adj)
# print(adj.shape)
# adj = sparse.csr_matrix(adj)
# print('adj', adj)
from collections import defaultdict
import numpy as np
# 19945
# graph = defaultdict(list)
# for i in range(0, 19945, 5):
#     for j in range(i, i+5):
#         if j != i:
#             graph[i].append(j)
#         if j != i+1:
#             graph[i+1].append(j)
#         if j != i+2:
#             graph[i+2].append(j)
#         if j != i+3:
#             graph[i+3].append(j)
#         if j != i+4:
#             graph[i+4].append(j)
#
# # print(graph)
# matrix = np.zeros((19945,2), dtype= int)
# for row in range(19945):
#     for col in range(5):
#         if row % 5 == col:
#             matrix[row][col] = 1

# print(matrix)
c = np.asmatrix(a)
fea = sparse.csr_matrix(c)
print(fea)
print(c)