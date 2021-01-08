from gcn.test import extract_matrix_v2
from gcn.test import predict_single
all, arr_so, arr_hoten, arr_ngaysinh, arr_quequan, arr_hktt = extract_matrix_v2.get_pre_data('../data_cmnd/text_test.txt', '../data_cmnd/pos_test.txt')
import numpy as np

arr_predict = predict_single.predict_label()
print(arr_predict)

# import glob
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

name_img = 'id_viettel_front_000616.jpg'
src = '/home/hq-lg/gcn/gcn/pictures/' + name_img
img = cv2.imread(src)
img = cv2.resize(img, (1060,673))

cv2.rectangle(img,(445,172),(503,215),(0,255,0),3)
cv2.rectangle(img,(342,261),(450,289),(0,255,0),3)
cv2.rectangle(img,(342,375),(494,415),(0,255,0),3)
cv2.rectangle(img,(342,435),(525,471),(0,255,0),3)
cv2.rectangle(img,(342,554),(645,584),(0,255,0),3)

_pos_0 = tuple(map(tuple, np.array([arr_predict[0]['pos_0'].astype(int)])))
cv2.putText(img,arr_predict[0]['labels_0'],_pos_0[0], font, 1,(255,0,0),2)

_pos_1 = tuple(map(tuple, np.array([arr_predict[0]['pos_1'].astype(int)])))
cv2.putText(img,arr_predict[0]['labels_1'],_pos_1[0], font, 1,(255,0,0),2)

_pos_2 = tuple(map(tuple, np.array([arr_predict[0]['pos_2'].astype(int)])))
cv2.putText(img,arr_predict[0]['labels_2'],_pos_2[0], font, 1,(255,0,0),2)

_pos_3 = tuple(map(tuple, np.array([arr_predict[0]['pos_3'].astype(int)])))
cv2.putText(img,arr_predict[0]['labels_3'],_pos_3[0], font, 1,(255,0,0),2)

_pos_4 = tuple(map(tuple, np.array([arr_predict[0]['pos_4'].astype(int)])))
cv2.putText(img,arr_predict[0]['labels_4'],_pos_4[0], font, 1,(255,0,0),2)

cv2.imshow('Display Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Create a black image
# for i in range(30):
#     src = '../pictures/id_viettel_front_00000{}.jpg'.format(i+1)
#     print(src)
#     img = cv.imread(src)
#     cv.rectangle(img,(384,0),(510,128),(0,255,0),3)
#     cv.imshow('Display Image', img)
#     # cv.waitKey(0)




# # src = '../pictures/id_viettel_front_000069.jpg'
# img = cv2.imread(src)
# img = cv2.resize(img, (1060,673))
# # font = cv2.FONT_HERSHEY_SIMPLEX
# cv2.rectangle(img,(445,172),(503,215),(0,255,0),3)
# cv2.putText(img,'so',(474,193), font, 1,(255,0,0),2)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# # cv2.destroyAllWindows()




# Draw a diagonal blue line with thickness of 5 px


# from flask import Flask
# app = Flask(__name__)
#
# @app.route('/')
# def hello_world():
#     return 'Hello, World!'