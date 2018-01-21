# 更改文件名字以获得不同实例的结果

f = open('CNN_result.txt','r')

arr = f.readlines()[1:]

labels = []
predict = []
# x = range(len(arr)//12)
for i in range(len(arr)):
    if i % 1800 == 0:
        temp_arr = arr[i].split('\t')
        l = float(temp_arr[0][1:-1])
        pre = float(temp_arr[1][1:-2])
        # print(l)
        # print(pre)
        labels.append(l)
        predict.append(pre)
# for line in arr:
#     temp_arr = line.split('\t')
#     l = float(temp_arr[0][1:-1])
#     pre = float(temp_arr[1][1:-2])
#     # print(l)
#     # print(pre)
#     labels.append(l)
#     predict.append(pre)
# print(len(labels))
# print(len(predict))
import matplotlib.pyplot as plt

plt.title('instance 6, weak simulation')
plt.xlabel('time')
plt.ylabel('Temp')
plt.xlim(0,max(len(labels),len(predict)))
plt.ylim(min(min(labels),min(predict))-0.5,max(max(labels),max(predict))+0.5)

plt.plot(range(len(labels)),labels,c='Orange',label='label value')
plt.plot(range(len(predict)),predict,c='Green',label='predict value')
plt.show()