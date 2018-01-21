# 该脚本生成用来绘图的数据

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from PIL import Image
import os
import random

labelFile = open('./img_info_2.txt')
lines = labelFile.readlines()
f_result = open('drawing.txt','w')
f_result.write('label')
f_result.write('\t')
f_result.write('predict')
f_result.write('\n')

def data_images(reshape=True,start_index=0):
    # 单张读入数据与温度标签
    if start_index == len(lines):
        return
    images = np.empty((1, 67500))
    labels = np.empty((1,1))
    temp_arr = lines[start_index].split('\t')
    print(temp_arr,'temp_arr')
    image = Image.open(temp_arr[0]).convert('RGB')
    arr = np.array(image)
    images[0] = arr.reshape((-1, 67500))
    labels[0] = np.array([float(temp_arr[1][:-2])])
    rows = 150
    cols = 150
    if reshape:
        return images.reshape(1, rows, cols, 3), np.array(labels, dtype=np.float32)
    else:
        return images, np.array(labels, dtype=np.float32)
# 保持神经网络的一致
tf.set_random_seed(1)
np.random.seed(1)

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
# tf.summary.histogram('/outputs', outputs)

with tf.name_scope('loss'):

# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) # compute cost
    loss = tf.losses.mean_squared_error(output,tf_y)
    # tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    train_op = tf.train.AdamOptimizer(0).minimize(loss)

# f = open('RNN_restore_part_result.txt', 'w')
# f2 = open('RNN_restore_TT.txt','w')
# plt.ion()
# plt.show()
# fog_info = []
# f.write('time,  loss per image')
# f.write('\n')

with tf.Session() as sess:
    saver = tf.train.Saver()
    saver.restore(sess,'./rnn_net/model.ckpt-10000')
    # writer = tf.summary.FileWriter('./rnn_test_log', sess.graph)  # write to file
    # merged = tf.summary.merge_all()

    for i in range(0,len(lines)-2,150):
        batch_xs, batch_ys = data_images(start_index=i, reshape=False)
        # _loss = sess.run(loss, feed_dict={tf_x: batch_xs, tf_y: batch_ys})
        # 这里的output就是我们需要的预测温度值
        _output,_,_,_ = sess.run([output,h_c, h_n,train_op], feed_dict={tf_x: batch_xs,tf_y:batch_ys})
        print(_output,'output')
        f_result.write(str(batch_ys))
        f_result.write('\t')
        f_result.write(str(_output[0]))
        f_result.write('\n')
f_result.close()