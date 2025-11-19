# _*_ coding:utf-8 _*_

import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv
import os
import shutil


label_dir = r'D:\BaiduNetdiskDownload\CK+\Emotion'
image_dir = r'D:\BaiduNetdiskDownload\CK+\cohn-kanade-images'
root_dir = r'D:\BaiduNetdiskDownload\CK+'
aam_dir = r'D:\BaiduNetdiskDownload\CK+\Landmarks'


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
    files = os.listdir(path)
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):
            position = path + '\\' + file
            f = open(position)
            line = f.readline()  # only one row
            str = int(float(line))
            return str


# 这个函数的作用是获取txt里头的点的坐标，并且只获取前27个，其中17-16需要倒过来
def get_point(path):
    x_16, y_16 = [], []
    x_10, y_10 = [], []
    count = 0
    with open(path,'r',encoding='UTF-8') as A:
        for eachline in A:
            tmp = eachline.split("   ")
            if count < 17:
                tmp[1] = tmp[1].split('+')
                base1, index1 = tmp[1][0], tmp[1][1]
                base1 = base1.split('e')[0]
                x_16.append(float(base1)*(10**int(index1)))

                tmp[2] = tmp[2].strip('\n')
                tmp[2] = tmp[2].split('+')
                base2, index2 = tmp[2][0], tmp[2][1]
                base2 = base2.split('e')[0]
                y_16.append(float(base2) * (10 ** int(index2)))
            elif count >= 17 and count <= 26:
                tmp[1] = tmp[1].split('+')
                base1, index1 = tmp[1][0], tmp[1][1]
                base1 = base1.split('e')[0]
                x_10.append(float(base1) * (10 ** int(index1)))

                tmp[2] = tmp[2].strip('\n')
                tmp[2] = tmp[2].split('+')
                base2, index2 = tmp[2][0], tmp[2][1]
                base2 = base2.split('e')[0]
                y_10.append(float(base2) * (10 ** int(index2)))

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
    plt.show()

# 用来测试点的坐标
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

# 保存图片，第一个是图片地址，第二个是图片的文件名，第三个是图片
def save_img(imgpath,img_name, img):
    print(imgpath)
    if not os.path.exists(imgpath):
        os.makedirs(imgpath)
        print(1)
    width = int(100)
    height = int(100)
    dim = (width, height)
    # 灰度化处理
    im_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # resize image
    resized = cv.resize(im_gray, dim, interpolation=cv.INTER_AREA)
    # 开始保存
    full_path = imgpath+"//"+img_name
    cv.imwrite(full_path, resized)

# 检查是否存在标签
def check_facs(file_name):
    if(os.path.exists(file_name) == False):
        return -1
    files = os.listdir(file_name)
    for file in files:  # 遍历文件夹
        if not os.path.isdir(file):
            position = file_name + '\\' + file
            f = open(position)
            line = f.readline()  # only one row
            if len(line) == 0:
                return -1
            else:
                return 1


# 循环获取图片的地址，如果标签为不为空则开始剪裁，这个函数是取所有的表情进行剪裁，适合做表情强度分析
# def process_pic(aampath):
#     for lapth in aampath:
#         fullpath = aam_dir + "//" + lapth
#         files = os.listdir(fullpath)  # 得到文件夹下的所有文件名称
#         for file in files:  # 遍历文件夹
#             if not os.path.isdir(file):  # 判断是否是文件夹，不是文件才打开
#                 aam_path = fullpath + "//" + file
#                 # 获取对应的图片的名字和对应的坐标文件的名字
#                 file_name =file.split("_landmarks")[0]
#                 t_img_path = image_dir +"//"+ lapth + "//" + file_name + ".png"
#                 # 对应的Label是否存在
#                 t_facs_path = label_dir + "//" + lapth
#                 is_facs = check_facs(t_facs_path)
#                 if is_facs == -1 or is_facs == None:
#                     continue
#                 point_list = get_point(aam_path)
#                 cut_result = ROI_byMouse(imgpath=t_img_path, lsPointsChoose=point_list)
#                 label_value = get_label(t_facs_path)
#                 save_path = root_dir + "//" + str(label_value)
#                 file_name = file.split("_landmarks")[0] + ".png"
#                 save_img(imgpath=save_path, img_name= file_name,img=cut_result)


# 循环获取图片的地址，如果标签为不为空则开始剪裁，这个函数是只取最后三个
def process_pic(aampath):
    for lapth in aampath:
        fullpath = aam_dir + "//" + lapth
        files = os.listdir(fullpath)  # 得到文件夹下的所有文件名称
        count = 0 #只取文件的最后三个，所以需要有一个计数器
        for i in range(0, len(files)):  # 遍历文件夹
            file = files[len(files)-count-1]
            count += 1
            if count > 3:
                break
            if not os.path.isdir(file):  # 判断是否是文件夹，不是文件才打开
                aam_path = fullpath + "//" + file
                # 获取对应的图片的名字和对应的坐标文件的名字
                file_name =file.split("_landmarks")[0]
                t_img_path = image_dir +"//"+ lapth + "//" + file_name + ".png"
                # 对应的Label是否存在
                t_facs_path = label_dir + "//" + lapth
                is_facs = check_facs(t_facs_path)
                if is_facs == -1 or is_facs == None:
                    continue
                point_list = get_point(aam_path)
                cut_result = ROI_byMouse(imgpath=t_img_path, lsPointsChoose=point_list)
                label_value = get_label(t_facs_path)
                save_path = root_dir + "//" + str(label_value)
                file_name = file.split("_landmarks")[0] + ".png"
                save_img(imgpath=save_path, img_name= file_name,img=cut_result)

# 主模块
if __name__ == "__main__":
    # 运行
    dir_list = get_label_path(aam_dir)
    process_pic(dir_list)


