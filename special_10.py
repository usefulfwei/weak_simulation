import xlrd
import os
from PIL import Image
import cv2
import numpy as np

T_info = {}

temp_arr = []
# form_path = r'./form/'

# for file in os.listdir(form_path):
    # pathTmp = os.path.join(form_path, file)
workbook = xlrd.open_workbook('./10.xlsx')
sheet = workbook.sheet_by_index(0)
result = ['./10.xlsx']
print('./10.xlsx','files')
for row in range(sheet.nrows):
    print(sheet.cell(row, 1).value)
    result.append(sheet.cell(row, 1).value)
# print(result)
temp_arr.append(result)
print(temp_arr)
for arr in temp_arr:
    # if arr[0][:1] not in T_info.keys():
    temp = []
    key = '10'
    print(key,'key')
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
f = open('img_10_cnn.txt', 'w')
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
# split_dict['10'] = [100,250, 820,970]
split_dict['10'] = [15,335, 735,1055]
split_dict['11'] = [100,420, 900,1220]
split_dict['12'] = [50,370, 850,1170]
split_dict['13'] = [50,370, 850,1170]
split_dict['14'] = [50,370, 850,1170]
split_dict['15'] = [100,420, 750,1170]
split_dict['16'] = [20,340, 750,1170]
split_dict['17'] = [20,340, 750,1170]
split_dict['18'] = [20,340, 750,1170]
split_dict['19'] = [50,370, 800,1120]
split_dict['20'] = [50,370, 800,1120]
split_dict['21'] = [50,370, 800,1120]

# for name in os.listdir(video_path):
# video_file = os.path.join(video_path, name)
# print(video_file, 'videoname')

labels = T_info['10']
index = 0
count = 0
cap = cv2.VideoCapture('./10.avi')
# while (cap.isOpened()):
# print('goes here')
ret, frame = cap.read()
while ret == True:
    # print('goes here')
    count += 1
    if count % 150 == 0:
        index += 1
    img = cv2.cvtColor(np.array(frame[15:335, 735:1055]),
                       cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    try:
        img.save('./img_320_cnn/'+str(amount)+'_10.jpg')
        f.write('./img_320_cnn/'+str(amount)+'_10.jpg')
        f.write('\t')
        f.write(str(labels[index]))
        f.write('\n')
        print('write one line')
    except:
        print('break')
        break
    ret, frame = cap.read()
    amount += 1

print(amount, 'the amount')
f.close()