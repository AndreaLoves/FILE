# encoding:utf-8
import string
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import cv2 as cv
import os.path
import glob

label_dir = r'D:\graduate\code\dataset\genki\labels.txt'
image_dir = r'D:\graduate\code\dataset\genki\imgaes'


def get_label(path):
    labels = []
    f = open(path)
    line = f.readline()
    while line:
        line = f.readline()
        if len(line)==0:
           break
        else:
            label = int(line[0])
            labels.append(label)

    f.close()
    return labels

def move_img(srcfile,dstpath):
    shutil.copy(srcfile, dstpath)  # 复制文件


def convert(img,from_path,to_path):
    srim1 = cv.imread(from_path+img)
    im2 = cv.resize(srim1,(100,100),)  # 为图片重新指定尺寸
    cv.imwrite(to_path+img,im2)



# 循环获取图片的地址，如果标签为不为空则开始剪裁
def process_pic(image_dir,labels):
    ls = os.listdir(image_dir)
    for index in range(len(labels)):
        temp = ls[index]
        if labels[index] == 0:
            srcfile = r"D:\graduate\code\dataset\genki\imgaes"+"/"+temp
            dstpath = r"D:\graduate\code\dataset\genki\0"
            move_img(srcfile,dstpath)
        else:
            srcfile = r"D:\graduate\code\dataset\genki\imgaes" + "/" + temp
            dstpath = r"D:\graduate\code\dataset\genki\1"
            move_img(srcfile, dstpath)



# 主模块
if __name__ == "__main__":
    # 运行
    # labels = get_label(label_dir)
    # process_pic(image_dir,labels)
    img_0_path = r"D:\graduate\code\dataset\genki\1"
    to_path = r"D:/graduate/code/dataset/genki/11"+ "/"
    from_path=r"D:/graduate/code/dataset/genki/1"+ "/"
    ls = os.listdir(img_0_path)
    for index in range(len(ls)):
        temp = ls[index]
        convert(temp, from_path, to_path)
    # from_path= ""
    # to_path= "./data/"
    # file = "file2162.jpg"
    # convert(file, from_path,to_path)

