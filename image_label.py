# -*- coding: utf-8 -*-
# python2
# 该脚本集合
#  存储图片，生成标签值并与图片做对应
#  并存储结果与img_info.txt中
# author:Lichang

import xlrd
import os
from PIL import Image
import cv2
import numpy as np


# 读取文档 温度值
T_info = {}

temp_arr = []
form_path = r'./form/'

for file in os.listdir(form_path):
    pathTmp = os.path.join(form_path, file)
    workbook = xlrd.open_workbook(pathTmp)
    sheet = workbook.sheet_by_index(0)
    result = [file]
    print(file,'files')
    for row in range(sheet.nrows):
        print(sheet.cell(row, 1).value)
        result.append(sheet.cell(row, 1).value)
    # print(result)
    temp_arr.append(result)
print(temp_arr)

# 插值法 11 点插 每 5 秒一个温度值
for arr in temp_arr:
    temp = []
    key = arr[0].split('.')[0]
    # print(key)
    for i in range(1, len(arr)-1):
        temp.append(arr[i])
        gap = round((arr[i+1] - arr[i]) / 11, 3)
        for j in range(1,12):
            temp.append(arr[i] + j*gap)
    temp.append(arr[len(arr)-1])
    T_info[key] = temp

print(T_info,'t_info')
print(T_info.keys(),'keys')
video_path = './video/'
f = open('img_info_320.txt', 'w')
amount = 1
'''
6 [200:350, 900:1050]
7 [150:300, 850:1000]
8 [200:350, 800:950]
9 [200:350, 900:1050]
10 [100:250, 820:970]
11 [150:300, 950:1100]
12 [80:230, 900:1050]
13 [80:230, 900:1050]
14 [80:230, 900:1050]
15 [200:350, 800:950]
16 [100:250, 800:950]
17 [100:250, 800:950]
18 [100:250, 800:950]
19 [100:250, 850:1000]
20 [50:200, 850:1000]
21 [100:250, 850:1000]
'''
split_dict = {}
split_dict['6'] = [100,420,800,1120]
split_dict['7'] = [50,370, 750,1070]
split_dict['8'] = [100,420, 700,1020]
split_dict['9'] = [100,420, 800,1120]
split_dict['10'] = [15,335, 735,1055]
split_dict['11'] = [100,420, 850,1170]
split_dict['12'] = [50,370, 850,1170]
split_dict['13'] = [50,370, 800,1120]
split_dict['14'] = [50,370, 850,1170]
split_dict['15'] = [100,420, 750,1070]
split_dict['16'] = [20,340, 750,1070]
split_dict['17'] = [20,340, 750,1070]
split_dict['18'] = [20,340, 750,1070]
split_dict['19'] = [50,370, 800,1120]
split_dict['20'] = [50,370, 800,1120]
split_dict['21'] = [50,370, 800,1120]

for name in os.listdir(video_path):
    video_file = os.path.join(video_path, name)
    print(video_file, 'videoname')
    video_code = name.split('.')[0]
    labels = T_info[video_code]
    f_v = open(str(video_code)+'_320.txt','w')
    index = 0
    count = 0
    cap = cv2.VideoCapture(video_file)
    ret, frame = cap.read()
    while ret == True:
        # print('goes here')
        count += 1
        # 每秒 30 帧 5秒一个温度，故 150 帧一个温度
        if count % 150 == 0:
            index += 1
        img = cv2.cvtColor(np.array(frame[split_dict[video_code][0]:split_dict[video_code][1],
                                    split_dict[video_code][2]:split_dict[video_code][3]]),
                           cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        img.save('./320_img/'+str(amount)+'_'+ str(video_code) +'.jpg')
        # 信息存储在 320_img.txt 中
        f.write('./320_img/'+str(amount)+'_'+ str(video_code) +'.jpg')
        f.write('\t')
        f.write(str(labels[index]))
        f.write('\n')
        f_v.write('./320_img/' + str(amount) + '_' + str(video_code) + '.jpg')
        f_v.write('\t')
        f_v.write(str(labels[index]))
        f_v.write('\n')
        # print('write one line')
        ret, frame = cap.read()
        amount += 1
    f_v.close()
print(amount, 'the amount')
f.close()