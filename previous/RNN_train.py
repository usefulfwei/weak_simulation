import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

'''
训练并生成网络
'''


def mean_square_loss_cal(arr1, arr2):
    val = 0
    for x in range(len(arr1)):
        val += (arr1[x]-arr2[x])**2
    return val/len(arr1)


from gen_data import *
test_size = 100
images, labels = test_data(size=test_size,reshape=False)

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
TIME_STEP = 150          # 图片的列数 rnn time step / image height
INPUT_SIZE = 150         # 图片的行数 rnn input size / image width

# 读取并变换输入维度 batch_size, time_step,input_size
with tf.name_scope('input'):
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE*3])/255.       # shape(batch, 784)
    image = tf.reshape(tf_x, [-1, TIME_STEP*3, INPUT_SIZE])                   # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None, 1])                             # input y

# RNN 调用RNN api
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=512)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
# 缩减维度为 1 --> 即所要求的温度
output = tf.layers.dense(outputs[:, -1, :], 1)              # output based on the last output step
# tensorboard 语句
# histogram 条形图
# scalar 折线图
tf.summary.histogram('/outputs', outputs)

with tf.name_scope('loss'):
# loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output) # compute cost
    loss = tf.losses.mean_squared_error(output,tf_y)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 优化器
    train_op = tf.train.AdamOptimizer(1e-5).minimize(loss)


init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) # the local var is for accuracy_op
# sess.run(init_op)     # initialize var in graph

# txt文本文档句柄
f = open('RNN_restore_result.txt', 'w')
f.write('time,  loss per image')
f.write('\n')

# matplot 绘图句柄
plt.ion()
plt.show()

# mat 文件句柄
fog_info = []



with tf.Session() as sess:
    # 激活变量 必须
    sess.run(tf.global_variables_initializer())
    # tensorboard 写入路径 './rnnlog'
    writer = tf.summary.FileWriter('./rnnlog', sess.graph)  # write to file
    merged = tf.summary.merge_all()  # operation to merge all summary
    # 保存模型
    saver = tf.train.Saver(tf.global_variables())
    for i in range(11200):
        # 调用gen_data.py 脚本 nums_image 为该批次数据的数量
        nums_image, batch_xs, batch_ys = data_images(start_index=int(i * 100), batch_size=100,reshape=False)
        # 训练模型 train_op
        sess.run(train_op, feed_dict={tf_x: batch_xs, tf_y: batch_ys})
        if i % 50 == 0:
            # 写入tensorboard
            result, _loss,_output = sess.run([merged, loss,output],
                                     feed_dict={tf_x: images, tf_y: labels})
            writer.add_summary(result, i)
            # 储存结果
            print(_loss)
            print(mean_square_loss_cal(batch_ys,_output),'my function')
            fog_info.append([i, _loss/test_size])
            f.write(str(i))
            f.write('\t')
            f.write(str(_loss/test_size))
            f.write('\n')
            plt.scatter(i, _loss/test_size, c='Red', s=30)
            plt.draw()
            plt.pause(0.1)
            if i%1000 == 0:
                saver.save(sess, './rnn_net/model.ckpt', global_step=i)

plt.savefig('RNN_restore.jpg')
sio.savemat('RNN_restore_fog.mat', {'fog_accuracy': fog_info})
f.close()
