# weak_simulation
毕设项目弱刺激部分代码（InceptionV3,tensorflow: CNN and RNN）

包含代码功能：

1. 将图片转化为tensorflow API所需要的数据格式，例如

全连接： batch_size,img_channel*width*height
CNN: batch_size,width,height,channels

2. 训练并生成模型以及相关你数据：

模型包括CNN
RNN（previous中上一版本，效果不理想弃用）
Inception V3，目前仍在训练中。

3. 利用生成模型，画出图标。结果在predict中，原实验有16个实例，为保护隐私，只展示单个实例的结果，
在predict（CNN）以及i_predict（InceptionV3）
