'''
分割结果
对应15 个实例
'''

f = open('drawing.txt','r')
arr = f.readlines()

txt_arr = [
    '11.txt',
    '12.txt',
    '13.txt',
    '14.txt',
    '15.txt',
    '16.txt',
    '17.txt',
    '18.txt',
    '19.txt',
    '20.txt',
    '21.txt',
    '6.txt',
    '7.txt',
    '8.txt',
    '9.txt',
    '10.txt',
]

for i in range(len(txt_arr)):
    f_txt = open(txt_arr[i],'w')
    if (i+1)*480 > len(arr):
        final_index = len(arr)
    else:
        final_index = (i+1)*480
    for j in range(i*480,final_index):
        f_txt.write(str(arr[j]))
    f_txt.close()
