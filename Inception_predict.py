import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
import os
from keras.models import load_model
from gen_data_CNN_sum import *

model = load_model('weak_simulation_1_40000_.h5')

def predict(instance_code):
    # model.summary()
    if not os.path.exists('./i_predict'):
        os.makedirs('./i_predict')
    f_c = open('./i_predict/'+str(instance_code)+'CNN_predict_result.txt','w')
    filename = str(instance_code) + '_320.txt'
    print(filename,'name')
    try:
        for i in range(0,150000,1800):
            nums_image, batch_xs, batch_ys = data_predict(filename,start_index=int(i), batch_size=1, reshape=True)
            # print(batch_xs)
            _output = model.predict(batch_xs)
            # print(_output)
            for x in range(len(batch_ys)):
                print(batch_ys[x], 'labels', _output[x], 'predict')
                f_c.write(str(batch_ys[x]))
                f_c.write('\t')
                f_c.write(str(_output[x]))
                f_c.write('\n')
    except:
        f_c.close()
    f_d = open('./i_predict/'+str(instance_code)+'CNN_predict_result.txt', 'r')
    arr = f_d.readlines()
    labels = []
    predict = []
    for i in range(len(arr)):
        # if i % 1800 == 0:
        temp_arr = arr[i].split('\t')
        l = float(temp_arr[0][1:-1])
        pre = float(temp_arr[1][1:-2])
        labels.append(l)
        predict.append(pre)
    import matplotlib.pyplot as plt
    plt.title('instance,'+str(instance_code)+' weak simulation')
    plt.xlabel('round')
    plt.ylabel('Temp')
    plt.xlim(0, max(len(labels), len(predict)))
    plt.ylim(min(min(labels), min(predict)) - 0.5, max(max(labels), max(predict)) + 0.5)

    plt.plot(range(len(labels)), labels, c='Orange', label='label value')
    plt.plot(range(len(predict)), predict, c='Green', label='predict value')
    plt.savefig('./i_predict/' + str(instance_code) + '.jpg')
    plt.cla()
    # plt.show()
for i in range(6,22):
    predict(i)