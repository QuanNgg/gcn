import tensorflow as tf
from gcn.models import GCN, MLP
from gcn.utils import *
from gcn.test import extract_matrix
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

# adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)
features, graph, y_train = extract_matrix.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
labels = y_train
idx_train = range(len(y_train))
train_mask = sample_mask(idx_train, labels.shape[0])

y_val = np.zeros(labels.shape)

idx_val = range(len(y_train))
val_mask = sample_mask(idx_val, labels.shape[0])
y_val[val_mask, :] = labels[val_mask, :]



features = preprocess_features(features)
if FLAGS.model == 'gcn':
    support = [preprocess_adj(adj)]
    num_supports = 1
    model_func = GCN
elif FLAGS.model == 'gcn_cheby':
    support = chebyshev_polynomials(adj, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree
    model_func = GCN
elif FLAGS.model == 'dense':
    support = [preprocess_adj(adj)]  # Not used
    num_supports = 1
    model_func = MLP
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
# print(outs, outs.shape)
predict = np.zeros((19945), dtype=int)
i = 0
for row in outs:
    max = 0
    j = 0
    for percent in row:
        if max < percent:
            max = percent
            predict[i] = j
        j+=1
    i+=1

true = np.zeros((19945), dtype=int)
for i_true in range(0, 19945, 5):
    true[i_true] = 0
    true[i_true+1] = 1
    true[i_true+2] = 2
    true[i_true+3] = 3
    true[i_true+4] = 4

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)
from sklearn.metrics import confusion_matrix
a= confusion_matrix(true, predict, labels=[0,1,2,3,4])
print(a)