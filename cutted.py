import cv2
import os
from PIL import Image
import numpy as np

'''
下面的数组表示手背的采集区域
实例标号 [区域范围]

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
video_file = os.path.join(r'./video/' + '21.avi')
cap = cv2.VideoCapture(video_file)
ret, frame = cap.read()
count = 305568
while ret == True:
    # 这一部分查看区域是否有效
    frame_r = frame[100:, 850:]
    img_test = cv2.cvtColor(frame_r, cv2.COLOR_BGR2RGB)
    img_test = Image.fromarray(img_test)
    img_test.show()
    ret, frame = cap.read()

    # width = len(frame_r)
    # height = len(frame_r[0])
    # for i in range(width // 48):
    #     if 48 * i + 48 > width:
    #         break
    #     else:
    #         for j in range(height // 48):
    #             if j * 48 + 48 > height:
    #                 break
    #             else:
    #                 try:
    #                     img = cv2.cvtColor(frame_r[48 * i:48 * i + 48, 48 * j:48 * j + 48], cv2.COLOR_BGR2RGB)
    #                     img = Image.fromarray(img)
    #                     img.save('./small_1/' + str(count) + '.jpg')
    #                     print('save one')
    #                     count += 1
    #                 except:
    #                     pass
    #         ret, frame = cap.read()

