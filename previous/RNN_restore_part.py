import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

from gen_data import *

# 该脚本用来恢复模型

tf.set_random_seed(1)
np.random.seed(1)

# 训练网络 即图与训练脚本（RNN_train.py） 保持一致
# Hyper Parameters
TIME_STEP = 150          # rnn time step / image height
INPUT_SIZE = 150         # rnn input size / image width

with tf.name_scope('input'):
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE*3])/255.       # shape(batch, 784)
    image = tf.reshape(tf_x, [-1, TIME_STEP*3, INPUT_SIZE])                   # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None, 1])                             # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 1)              # output based on the last output step
tf.summary.histogram('/outputs', outputs)

with tf.name_scope('loss'):

# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) # compute cost
    loss = tf.losses.mean_squared_error(output,tf_y)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)

#结果储存句柄
f = open('RNN_restore_part_result.txt', 'w')
plt.ion()
plt.show()
fog_info = []
f.write('time,  loss per image')
f.write('\n')

with tf.Session() as sess:
    # 恢复模型
    saver = tf.train.Saver()
    saver.restore(sess,'./rnn_net/model.ckpt-100000')
    writer = tf.summary.FileWriter('./rnn_test_log', sess.graph)  # write to file
    merged = tf.summary.merge_all()

    for i in range(0,130001,50):
        nums_image, batch_xs, batch_ys = data_images(start_index=int(i * 10), batch_size=10, reshape=False)
        # 这里不再运行train_op 只关心loss
        _loss = sess.run(loss, feed_dict={tf_x: batch_xs, tf_y: batch_ys})
        print(_loss/nums_image)
        fog_info.append([i, _loss / nums_image])
        f.write(str(i))
        f.write('\t')
        f.write(str(_loss / nums_image))
        f.write('\n')
        plt.scatter(i, _loss / nums_image, c='Red', s=100)
        plt.draw()
        plt.pause(0.1)

plt.savefig('RNN_restore_part_result.jpg')
sio.savemat('RNN_restore_part_result.mat', {'fog_accuracy': fog_info})
f.close()