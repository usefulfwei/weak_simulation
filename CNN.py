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

from gen_data_CNN import *
test_size = 50
images, labels = test_data(size=test_size,reshape=False)
print(images.shape,'image')
print(labels.shape,'label')

def conv2d(x, W):
    # stride [1, x_movement, y_movement, 1]
    # Must have strides[0] = strides[3] = 1
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def conv2d_with_batch_normal(x, w):
    tf.nn.batch_norm_with_global_normalization()
    pass

def max_pool_2x2(x):
    # stride [1, x_movement, y_movement, 1]
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


# Hyper Parameters
TIME_STEP = 320          # 图片的列数 rnn time step / image height
INPUT_SIZE = 320         # 图片的行数 rnn input size / image width

# 读取并变换输入维度 batch_size, time_step,input_size
with tf.name_scope('input'):
    tf_x = tf.placeholder(tf.float32, [None, TIME_STEP * INPUT_SIZE*3])/255.       # shape(batch, 784)
    image = tf.reshape(tf_x, [-1, TIME_STEP, INPUT_SIZE,3])                   # (batch, height, width, channel)
    tf_y = tf.placeholder(tf.float32, [None, 1])                             # input y


W_conv1 = tf.Variable(tf.truncated_normal([5,5, 3,32], stddev=0.1),name='w1') # patch 5x5, in size 1, out size 32
b_conv1 = tf.Variable(tf.constant(0.1, shape=[32]) ,name='b1')
h_conv1 = tf.nn.relu(conv2d(image, W_conv1) + b_conv1) # output size 28x28x32
h_pool1 = max_pool_2x2(h_conv1)
# output size 14x14x32


W_conv2 = tf.Variable(tf.truncated_normal([3,3, 32,64], stddev=0.1),name='w2') # patch 5x5, in size 32, out size 64
b_conv2 = tf.Variable(tf.constant(0.1,shape=[64]),name='b2')
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2) # output size 14x14x64
h_pool2 = max_pool_2x2(h_conv2)
# output size 7x7x64

W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 64, 256], stddev=0.1),
                      name='w2')  # patch 5x5, in size 32, out size 64
b_conv3 = tf.Variable(tf.constant(0.1, shape=[256]), name='b2')
h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)  # output size 14x14x64
h_pool3 = max_pool_2x2(h_conv3)

W_fc1 = tf.Variable(tf.truncated_normal([40*40*256, 64], stddev=0.1),name='w3')
b_fc1 = tf.Variable(tf.constant(0.1,shape=[64]),name='b3')
h_pool3_flat = tf.reshape(h_pool3, [-1, 40*40*256])
h_fc1 = tf.nn.relu(tf.matmul(h_pool3_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, 0.5)

output = tf.layers.dense(h_fc1_drop, 1)
# tensorboard 语句
# histogram 条形图
# scalar 折线图

with tf.name_scope('loss'):
    loss = tf.losses.absolute_difference(labels=tf_y,predictions=output)
    tf.summary.scalar('loss', loss)
with tf.name_scope('train'):
    # 优化器
    train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)


def train():
    f = open('cnn_result.txt', 'w')
    f.write('time,  loss per image')
    f.write('\n')
    # matplot 绘图句柄
    plt.ion()
    plt.show()

    # mat 文件句柄
    fog_info = []

    with tf.Session() as sess:
    #     # 激活变量 必须
        saver = tf.train.Saver()
        saver.restore(sess, './cnn_net/model.ckpt-39')
    #
        # sess.run(tf.global_variables_initializer())
    #     # tensorboard 写入路径 './rnnlog'
        writer = tf.summary.FileWriter('./cnnlog', sess.graph)  # write to file
        merged = tf.summary.merge_all()  # operation to merge all summary
    #     # 保存模型
        saver = tf.train.Saver(tf.global_variables())
        for epoch in range(40,50):
            # 11200
            for i in range(618*10):
                # 调用gen_data.py 脚本 nums_image 为该批次数据的数量
                nums_image, batch_xs, batch_ys = data_images(start_index=int(i * 10), batch_size=10,reshape=False)
                # 训练模型 train_op
                sess.run(train_op, feed_dict={tf_x: batch_xs, tf_y: batch_ys})
                if i % 50 == 0:
                    # 写入tensorboard
                    result, _loss,_output = sess.run([merged, loss,output],feed_dict={tf_x: images, tf_y: labels})
                    writer.add_summary(result, i)
                    # 储存结果
                    for x in range(len(images)):
                        print(images[x],'labels',_output[x],'predict')
                    print(_loss)
                    # print(mean_square_loss_cal(batch_ys,_output),'my function')
                    fog_info.append([i, _loss])
                    f.write(str(i))
                    f.write('\t')
                    f.write(str(_loss))
                    f.write('\n')
                    plt.scatter(i, _loss*10, c='Red', s=30)
                    plt.draw()
                    plt.pause(0.1)
            saver.save(sess, './cnn_net/model.ckpt', global_step=epoch)

    plt.savefig('cnn_net.jpg')
    sio.savemat('cnn_net.mat', {'fog_accuracy': fog_info})
    f.close()


def predict():
    f_c = open('CNN_result.txt','w')
    with tf.Session() as sess:
        #     # 激活变量 必须
        saver = tf.train.Saver()
        saver.restore(sess, './cnn_net/model.ckpt-39')
        for i in range(618 * 10):
            # 调用gen_data.py 脚本 nums_image 为该批次数据的数量
            nums_image, batch_xs, batch_ys = data_images(start_index=int(i * 10), batch_size=10, reshape=False,shuffed = False)
            # 训练模型 train_op
            _output = sess.run(output, feed_dict={tf_x: batch_xs})
            for x in range(len(batch_ys)):
                print(batch_ys[x], 'labels', _output[x], 'predict')
                f_c.write(str(batch_ys[x]))
                f_c.write('\t')
                f_c.write(str(_output[x]))
                f_c.write('\n')
        f_c.close()

predict()