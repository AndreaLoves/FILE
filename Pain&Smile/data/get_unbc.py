# encoding:utf-8
import string
import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import cv2 as cv

label_dir = r'D:\graduate\code\pain\PSPI'
image_dir = r'D:\graduate\code\pain\Images'
facs_dir = r'D:\graduate\code\pain\FACS'
aam_dir = r'D:\graduate\code\pain\AAM_landmarks'
save_path_zero = r'D:\graduate\code\pain\0'
save_path_one = r'D:\graduate\code\pain\1'

dir_list = []

# 获取每个文件的地址，例如[120-kz120\\kz120t2aaunaff]对文件夹进行循环就好了
def get_label_path(pt):
    dir_list = []
    for root, dirs, _ in os.walk(pt):
        for rdir in dirs:
            for _, sub_dirs, _ in os.walk(root + '\\' + rdir):
                for sub_dir in sub_dirs:
                    # print(sub_dir)
                    dir_list.append(rdir + '\\' + sub_dir)
                    # print(rdir + '\\' + sub_dir)
                break
        break
    return dir_list


# 如果标签为空则返回-1，为0返回0，否则返回1
def get_label(path):
    f = open(path, 'r+')
    line = f.readline()  # only one row
    str = int(float(line))
    if str == 0:
        return 0
    elif str >0 and str <17:
        return 1
    else:
        return -1


# 这个函数的作用是获取txt里头的点的坐标，并且只获取前27个，其中17-16需要倒过来
def get_point(path):
    x_16, y_16 = [], []
    x_10, y_10 = [], []
    count = 0
    with open(path) as A:
        for eachline in A:
            tmp = eachline.split(" ")
            if count < 17:
                x_16.append(float(tmp[0]))
                y_16.append(float(tmp[1]))
            elif count >= 17 and count <= 26:
                x_10.append(float(tmp[0]))
                y_10.append(float(tmp[1]))
            else:
                break
            count = count + 1
    for i in range(0, 9):
        x_16.append(x_10[9 - i])
        y_16.append(y_10[9 - i])

    if x_16 == []:
        return 1
    else:
        point_list = []
        for temp in range(0, 26):
            point = [int(x_16[temp]), int(y_16[temp])]
            point_list.append(point)
        return point_list


# 这个函数的作用是通过点的坐标来裁剪图片
# opencv不规则裁剪
def ROI_byMouse(imgpath, lsPointsChoose):
    img = cv.imread(imgpath)
    mask = np.zeros(img.shape, np.uint8)
    pts = np.array(lsPointsChoose, np.int32)  # pts是多边形的顶点列表（顶点集）
    col0 = pts[:, 0]
    col1 = pts[:, 1]
    x1 = np.min(col0)
    y1 = np.min(col1)
    x2 = np.max(col0)
    y2 = np.max(col1)
    pts = pts.reshape((-1, 1, 2))
    # 这里 reshape 的第一个参数为-pain, 表明这一维的长度是根据后面的维度的计算出来的。
    # OpenCV中需要先将多边形的顶点坐标变成顶点数×pain×2维的矩阵，再来绘制

    # --------------画多边形---------------------
    mask = cv.polylines(mask, [pts], True, (255, 255, 255))
    # -------------填充多边形---------------------
    mask2 = cv.fillPoly(mask, [pts], (255, 255, 255))
    ROI = cv.bitwise_and(mask2, img)
    return ROI[y1:y2, x1:x2]


# 显示关键点的轨迹
def show_point(point):
    x = point[0]
    y = point[1]
    plt.scatter(x, y)
    plt.rcParams.update({'figure.figsize': (10, 8), 'figure.dpi': 100})
    # plt.title('Simple Scatter plot')
    # plt.xlabel('X - value')
    # plt.ylabel('Y - value')
    plt.show()


def draw_point(img, points_list):
    point_size = 1
    point_color = (0, 0, 255)  # BGR
    thickness = 4  # 可以为 0 、4、8

    # 要画的点的坐标

    for point in points_list:
        cv.circle(img, point, point_size, point_color, thickness)

    cv.namedWindow("image")
    cv.imshow('image', img)
    cv.waitKey(10000000)  # 显示 10000 ms 即 10s 后消失
    cv.destroyAllWindows()


def save_img(imgpath, img):
    width = int(100)
    height = int(100)
    dim = (width, height)
    # 灰度化处理
    im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # resize image
    resized = cv.resize(im_gray, dim, interpolation=cv.INTER_AREA)
    # 开始保存
    cv.imwrite(imgpath, resized)

def check_facs(file_name):
    f = open(file_name, 'r+')
    line = f.readline()  # only one row
    if len(line) == 0:
        return -1
    else:
        return 1

# 循环获取图片的地址，如果标签为不为空则开始剪裁
def process_pic(labelpath):
    for lapth in labelpath:
        fullpath = label_dir + "//" + lapth
        files = os.listdir(fullpath)  # 得到文件夹下的所有文件名称
        for file in files:  # 遍历文件夹
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件夹才打开
                label_value = get_label(fullpath + "//" + file)
                # 获取对应的图片的名字和对应的坐标文件的名字
                file_name =file.split("_facs")[0]
                t_aam_path =aam_dir+ "//"+ lapth + "//" + file_name + "_aam.txt"
                t_img_path = image_dir +"//"+ lapth + "//" + file_name + ".png"
                t_facs_path = facs_dir + "//" + lapth + "//" + file_name + "_facs.txt"
                is_facs = check_facs(t_facs_path)
                if is_facs == -1:
                    print(t_facs_path)
                    print(is_facs)
                    continue
                point_list = get_point(t_aam_path)
                cut_result = ROI_byMouse(imgpath=t_img_path, lsPointsChoose=point_list)
                if label_value == 0:
                    save_path = save_path_zero + "//" +file.split("_facs")[0] + ".png"
                    save_img(imgpath=save_path, img=cut_result)
                elif label_value == 1:
                    save_path = save_path_one + "//" + file.split("_facs")[0] + ".png"
                    save_img(imgpath=save_path, img=cut_result)


# 主模块
if __name__ == "__main__":
    # 运行
    dir_list = get_label_path(label_dir)
    process_pic(dir_list)


