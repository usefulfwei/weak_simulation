# -*- coding: utf-8 -*-

# author:Lichang
import os
import numpy
import random
from PIL import Image

# 读取img_info 转化为tensorflow所需要的图片格式

labelFile = open('./img_info_320.txt','r')
lines = labelFile.readlines()


# 图片翻转强化
def flip(images):
    arr = numpy.array(images)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    r1, g1, b1 = numpy.rot90(r), numpy.rot90(g), numpy.rot90(b)
    r2, g2, b2 = numpy.rot90(r,2), numpy.rot90(g,2), numpy.rot90(b,2)
    r3, g3, b3 = numpy.rot90(r,3), numpy.rot90(g,3), numpy.rot90(b,3)
    arr2 = numpy.dstack([r1, g1, b1])
    arr3 = numpy.dstack([r2, g2, b2])
    arr4 = numpy.dstack([r3, g3, b3])
    return numpy.array([arr,arr2,arr3,arr4])

def data_test(size=50, reshape=True):
    test_lines = lines[:]
    _index = [random.randint(0,len(test_lines)) for _ in range(size)]

    # random.shuffle(test_lines)
    # 图片 150*150 RGB三通道，总共有67500个像素点
    images = numpy.empty((size, 320*320*3))
    labels = numpy.empty((size,1))
    for i in range(size):
        temp_arr = test_lines[_index[i]].split('\t')
        image = Image.open(temp_arr[0]).convert('RGB')
        arr = numpy.array(image)
        images[i] = arr.reshape((-1,320*320*3))
        labels[i] = numpy.array([float(temp_arr[1][:-1])])
    print("done")
    num_images = size
    rows = 320
    cols = 320
    if reshape == True:
        return images.reshape(num_images, rows, cols, 3), numpy.array(labels, dtype=numpy.float32)
    else:
        return images, numpy.array(labels, dtype=numpy.float32)

def data_train(reshape=True,batch_size = 100,start_index=0,shuffed=True):
    train_lines = lines[:]
    if shuffed:
        random.shuffle(train_lines)
    if start_index+batch_size > len(train_lines):
        end_index = len(lines)
    elif start_index > len(lines):
        print('finish')
        return False
    else:
        end_index = start_index+batch_size
    images = numpy.empty((int(end_index-start_index), 320*320*3))
    labels = numpy.empty((int(end_index-start_index),1))
    amount = int((end_index-start_index))
    for i in range(start_index,end_index):
        temp_arr = train_lines[i].split('\t')
        image = Image.open(temp_arr[0]).convert('RGB')
        # 这里使用了数据的强化
        arr = numpy.array(image)
        images[i-start_index] = arr.reshape((-1, 320*320*3))
        labels[i-start_index] = numpy.array([float(temp_arr[1][:-1])])
    print('done')
    num_images = amount
    rows = 320
    cols = 320
    if reshape:
        return num_images,images.reshape(num_images, rows, cols, 3), numpy.array(labels, dtype=numpy.float32)
    else:
        return num_images,images, numpy.array(labels, dtype=numpy.float32)

def data_predict(filename, reshape=True,batch_size = 100,start_index=0):
    filepath = './individual_txt/'
    f_p = open(filepath+filename,'r')
    predict_line = f_p.readlines()
    if start_index+batch_size > len(predict_line):
        end_index = len(predict_line)
    elif start_index > len(predict_line):
        print('finish')
        return False
    else:
        end_index = start_index+batch_size
    images = numpy.empty((int(end_index-start_index), 320*320*3))
    labels = numpy.empty((int(end_index-start_index),1))
    amount = int((end_index-start_index))
    for i in range(start_index,end_index):
        temp_arr = predict_line[i].split('\t')
        image = Image.open(temp_arr[0]).convert('RGB')
        # 这里使用了数据的强化
        arr = numpy.array(image)
        images[i-start_index] = arr.reshape((-1, 320*320*3))
        labels[i-start_index] = numpy.array([float(temp_arr[1][:-1])])
    print('done')
    num_images = amount
    rows = 320
    cols = 320
    if reshape:
        return num_images,images.reshape(num_images, rows, cols, 3), numpy.array(labels, dtype=numpy.float32)
    else:
        return num_images,images, numpy.array(labels, dtype=numpy.float32)

