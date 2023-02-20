# 生成训练数据
import os
import numpy as np
import cv2

f_list_path = './dataset/Protocol/Pair_list_F.txt'
p_list_path = './dataset/Protocol/Pair_list_P.txt'

f1 = open(f_list_path,'r').read().split()
f2 = open(p_list_path,'r').read().split()
all_file_list = f1+f2
all_file_list = all_file_list[1::2]
with open('data_list.txt','w') as f:
    for line in all_file_list:
        data = []
        line = line[2:]
        if line.find('frontal') != -1:
            pose_label = 1
        else:
            pose_label = 0
        id_label = int(line[13:16])

        f.write(line+'\n')


f = open('data_list.txt','r')
# for line in f.readlines():
#     print(line.strip())
print(f.readlines()[::2])