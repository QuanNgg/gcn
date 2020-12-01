# from gcn.test import extract_matrix
import numpy as np
import json
from collections import defaultdict
# def load_input():
with open('../extract_matrix/graph.txt') as f:
    tmp = f.read()
    f.close()
tmp = json.loads(tmp)
graph = defaultdict(int,tmp)
# print(graph)
with open('../extract_matrix/labels_ally.txt') as f:
    tmp = f.read()
    f.close()
tmp = tmp.replace("[", "")
tmp = tmp.replace("]", "")
tmps = tmp.split("\n")
labels = np.zeros(shape=(19945,5))
for row in tmps:
    row = row.strip()
    numbers = row.split(' ')
a = np.array([1,3])
b = np.array([3,4])
a= np.vstack((a, b))
print(a)