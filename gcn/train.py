from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

import matplotlib.pyplot as plt

from gcn.utils import *
from gcn.models import GCN, MLP
from gcn.test import extract_matrix

import numpy
numpy.set_printoptions(threshold=sys.maxsize)

def sample_mask(idx, l):
    """Create mask."""
    mask = np.zeros(l)
    mask[idx] = 1
    return np.array(mask, dtype=np.bool)


# Set random seed
seed = 123
np.random.seed(seed)
tf.compat.v1.set_random_seed(seed)

# Settings
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

# Load data
adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data()
# print(y_val)
# print(shape=tf.constant(features[2], dtype=tf.int64))

# """

# Some preprocessing
features = preprocess_features(features)

# features, graph, y_train = input.get_data_from_file('./raw/text_1.txt', './raw/pos_1.txt')
# adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))
# labels = y_train
# idx_train = range(len(y_train))
# train_mask = sample_mask(idx_train, labels.shape[0])
#
# y_val = np.zeros(labels.shape)
#
# idx_val = range(len(y_train))
# val_mask = sample_mask(idx_val, labels.shape[0])
# y_val[val_mask, :] = labels[val_mask, :]
# features = preprocess_features(features)

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
    'support': [tf.compat.v1.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.compat.v1.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
    'labels': tf.compat.v1.placeholder(tf.float32, shape=(None, y_train.shape[1])),
    'labels_mask': tf.compat.v1.placeholder(tf.int32),
    'dropout': tf.compat.v1.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.compat.v1.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = model_func(placeholders, input_dim=features[2][1], logging=True)

# Initialize session
sess = tf.compat.v1.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.compat.v1.global_variables_initializer())

cost_val = []
epoch_t = []
train_acc = []
# Train model
for epoch in range(25000):
    epoch_t.append(epoch)

    t = time.time()
    # Construct feed dictionary
    feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
    feed_dict.update({placeholders['dropout']: FLAGS.dropout})

    # Training step
    outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)

    # Validation
    cost, acc, duration = evaluate(features, support, y_val, val_mask, placeholders)
    cost_val.append(cost)

    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]), "val_loss=", "{:.5f}".format(cost),
          "val_acc=", "{:.5f}".format(acc), "time=", "{:.5f}".format(time.time() - t))
    train_acc.append("{:.5f}".format(outs[2]))
    # if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
    #     print("Early stopping...")
    #     break

print("Optimization Finished!")

# Testing
test_cost, test_acc, test_duration = evaluate(features, support, y_test, test_mask, placeholders)
print("Test set results:", "cost=", "{:.5f}".format(test_cost),
      "accuracy=", "{:.5f}".format(test_acc), "time=", "{:.5f}".format(test_duration))

save = model.save(sess)
sess.close()

plt.xlabel('epoch')
plt.ylabel('train accury')
plt.plot(epoch_t, train_acc)
plt.show()

# """

# feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
# feed_dict.update({placeholders['dropout']: FLAGS.dropout})
# outs = sess.run(model.predict(), feed_dict=feed_dict)
# print(outs, outs.shape)

# outs = sess.run([model.opt_op, model.loss, model.accuracy, model.predict()], feed_dict=feed_dict)
# Epoch: 0200 train_loss= 0.56547 train_acc= 0.97857 val_loss= 1.04744 val_acc= 0.78600 time= 0.24111 vs model.predict()
# ----2500 line vs 500 epoch: Epoch: 0500 train_loss= 1.31984 train_acc= 0.20000 val_loss= 1.21317 val_acc= 0.60000 time= 2.78532
# + Test set results: cost= 1.21772 accuracy= 0.60000 time= 1.24332
# ----1500 num_line vs 1000 epoch: Epoch: 1000 train_loss= 1.13205 train_acc= 0.27939 val_loss= 1.06709 val_acc= 0.80000 time= 0.95047
# +Test set results: cost= 1.06709 accuracy= 0.80000 time= 0.38039
# ---- 1500 num_line vs 4000 epoch: Epoch: 4000 train_loss= 0.86916 train_acc= 0.64886 val_loss= 0.77561 val_acc= 0.92061 time= 0.97659
#+ Test set results: cost= 0.77561 accuracy= 0.92061 time= 0.39368
# ----1500 num_line vs 6000 epoch Epoch: 6000 train_loss= 0.74351 train_acc= 0.92061 val_loss= 0.69441 val_acc= 0.96031 time= 0.96522
# + Test set results: cost= 0.69441 accuracy= 0.96031 time= 0.39004
# ***----6000 num_line vs 8000 epoch: Epoch: 8000 train_loss= 0.97846 train_acc= 0.43871 val_loss= 0.80331 val_acc= 0.76129 time= 0.81284
# + Test set results: cost= 0.80331 accuracy= 0.76129 time= 0.25455
# ***----8000 num_line vs 8000 epoch: Epoch: 8000 train_loss= 0.84546 train_acc= 0.76302 val_loss= 0.83637 val_acc= 0.76302 time= 1.08696
# + Test set results: cost= 0.83637 accuracy= 0.76302 time= 0.44519
# ***---10000 num_line vs 10000epoch: Epoch: 10000 train_loss= 0.73202 train_acc= 0.92715 val_loss= 0.80397 val_acc= 0.76357 time= 1.54459
# + Test set results: cost= 0.80397 accuracy= 0.76357 time= 0.61993
#$$$ ---- 6000 num_line vs 15000 epoch: Epoch: 15000 train_loss= 0.75147 train_acc= 0.76129 val_loss= 0.68144 val_acc= 0.76129 time= 0.55241
# + Test set results: cost= 0.68144 accuracy= 0.76129 time= 0.22044
# **** 8000 num_line vs 15000 epoch: Epoch: 15000 train_loss= 0.81625 train_acc= 0.76302 val_loss= 0.68285 val_acc= 0.76302 time= 1.00483
# Test set results: cost= 0.68285 accuracy= 0.76302 time= 0.40359
# (5255, 5)
# (8000 numline + 5000 epoch)Epoch: 5000 train_loss= 0.75054 train_acc= 0.70498 val_loss= 0.75336 val_acc= 0.74595 time= 1.22704
# + Test set results: cost= 0.75336 accuracy= 0.74595 time= 0.48632
# (8000 num_line + old finetuning + 10000epoch)Epoch: 10000 train_loss= 0.70743 train_acc= 0.76656 val_loss= 0.64367 val_acc= 0.71263 time= 1.68311
# +Test set results: cost= 0.64367 accuracy= 0.71263 time= 0.66769
# 8k num_line (new finetuning) + 10k epoch: Epoch: 10000 train_loss= 0.66206 train_acc= 0.70500 val_loss= 0.58400 val_acc= 0.78185 time= 1.09991
# +Test set results: cost= 0.58400 accuracy= 0.78185 time= 0.45037
# Epoch: 10000 train_loss= 0.58391 train_acc= 0.74149 val_loss= 0.62862 val_acc= 0.73075 time= 1.08936
# +Test set results: cost= 0.62862 accuracy= 0.73075 time= 0.43689
# Epoch: 10000 train_loss= 0.72610 train_acc= 0.67864 val_loss= 0.65042 val_acc= 0.90220 time= 0.29688
# + Test set results: cost= 0.65042 accuracy= 0.90220 time= 0.11172
# Epoch: 15000 train_loss= 0.58134 train_acc= 0.89558 val_loss= 0.59764 val_acc= 0.96386 time= 0.34069
# + Test set results: cost= 0.59764 accuracy= 0.96386 time= 0.13535
# No finetuning pos [0] 8k + 20K Epoch: 20000 train_loss= 0.61675 train_acc= 0.75059 val_loss= 0.58411 val_acc= 0.77940 time= 0.53235
# + Test set results: cost= 0.58411 accuracy= 0.77940 time= 0.20435
# finetuning 5k + 20K poch: 20000 train_loss= 0.63648 train_acc= 0.85035 val_loss= 0.59778 val_acc= 0.96259 time= 0.35652
# + Test set results: cost= 0.59778 accuracy= 0.96259 time= 0.13890