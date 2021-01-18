import tensorflow as tf
from gcn.models import GCN, MLP
from gcn.utils import *
from gcn.test import extract_matrix
from gcn.test import extract_matrix_v2


import numpy
numpy.set_printoptions(threshold=sys.maxsize)
# features, graph, y_train = input.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
# adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))


flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 200, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')

def predict_label():
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(1)
    feat = features.toarray()
    print(len(feat))
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(1)],
        'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.compat.v1.placeholder(tf.int32),
        'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    sess = tf.compat.v1.Session()
    sess.run(tf.compat.v1.global_variables_initializer())

    model = GCN(placeholders, input_dim=features[2][1], logging=True)
    model.load(sess)


    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})
    outs = sess.run(model.predict(), feed_dict=feed_dict)

    # print(outs)
    # print(outs.shape)
    size = outs.shape[0]
    print('size', size)

    # all, so_size, hoten_size, ngaysinh_size, quequan_size, hktt_size = extract_matrix_v2.get_pre_data('/home/hq-lg/gcn/gcn/data_cmnd/text_test.txt', '/home/hq-lg/gcn/gcn/data_cmnd/pos_test.txt')

    predict = np.zeros((size), dtype=int)
    i = 0
    for row in outs:
        max_per = max(row)
        index = np.where(row == max_per)[0][0]
        predict[i] = index
        i += 1

    arr_predict = []
    for a in range(size-5, size, 5):
        arr_predict.append({
            'pos_0': feat[a],
            'labels_0': convert_num_to_string(predict[a]),
            'pos_1': feat[a+1],
            'labels_1': convert_num_to_string(predict[a+1]),
            'pos_2': feat[a+2],
            'labels_2': convert_num_to_string(predict[a+2]),
            'pos_3': feat[a+3],
            'labels_3': convert_num_to_string(predict[a+3]),
            'pos_4': feat[a+4],
            'labels_4': convert_num_to_string(predict[a+4])
        })

    return arr_predict


def convert_num_to_string(predict):
    if predict == 0:
        label = 'So cmt'
    elif predict == 1:
        label = 'Ho ten'
    elif predict == 2:
        label = 'Ngay sinh'
    elif predict == 3:
        label = 'Nguyen quan'
    elif predict == 4:
        label = 'Noi DKHK thuong tru'
    return label
