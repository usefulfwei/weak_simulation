# -*- coding: utf-8 -*-

# author:Lichang

import keras
from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D
from keras import backend as K
import matplotlib.pyplot as plt
from keras.callbacks import TensorBoard

# create the base pre-trained model
base_model = InceptionV3(weights='imagenet', include_top=False)

base_model.summary()

# add a global spatial average pooling layer
x = base_model.output
x = GlobalAveragePooling2D()(x)
# let's add a fully-connected layer
x = Dense(1024, activation='relu')(x)
# and a logistic layer -- let's say we have 200 classes
x = Dense(512, activation='relu')(x)
predictions = Dense(1, activation='relu')(x)

# this is the model we will train
model = Model(input=base_model.input, output=predictions)

adm = keras.optimizers.Adam(lr=0.001,beta_1=0.9,beta_2=0.999,epsilon=1e-08)

model.compile(optimizer=adm, loss='mae')

from gen_data_CNN_sum import *
test_size = 50
images, labels = data_test(size=test_size,reshape=True)


for epoch in range(50):
    # for epoch in range(start_point,50):
        # 11200
    f = open('./i_round/'+str(epoch)+'cnn_result.txt', 'w')
    f.write('time,  loss per image')
    f.write('\n')
    # matplot 绘图句柄
    plt.ion()
    plt.show()
    for i in range(0,1141860//20):
        # 调用gen_data.py 脚本 nums_image 为该批次数据的数量
        nums_image, batch_xs, batch_ys = data_train(start_index=int(i * 20), batch_size=20,reshape=True)
        model.train_on_batch(batch_xs,batch_ys)
        if i % 100 == 0:
            scores = model.evaluate(images,labels)
            print(scores)
            _predict = model.predict(images)
            for x in range(len(images)):
                print(labels[x],'labels',_predict[x],'predict')
            print(i,'goes here')
            f.write(str(i))
            f.write('\t')
            f.write(str(scores))
            f.write('\n')
            plt.scatter(i, scores*10, c='Red', s=20)
            plt.draw()
            plt.pause(0.1)
        if i % 10000 == 0:
            model.save('weak_simulation'+'_'+str(epoch)+'_'+str(i)+'_'+'.h5')
