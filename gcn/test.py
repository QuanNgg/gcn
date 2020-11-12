from gcn import input
import numpy as np
from scipy import sparse
a = input.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
# print((a))
aa = np.zeros((11))
# print(aa)
b = np.array(a, dtype=float)
# print(b)
fea = sparse.csr_matrix(b)
print('features', fea)
adj = []
for i in range(11):
    print(b[i])
    k = []
    for j in range(11):
        a = np.linalg.norm(b[i] - b[j])
        adj.append(a)
adj = np.array(adj, dtype=int)
adj = adj.reshape(11,11)
print(adj)
print(adj.shape)
adj = sparse.csr_matrix(adj)
print('adj', adj)