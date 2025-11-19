import csv
import os
import numpy as np
import h5py
import skimage.io

ck_path = r"D:/graduate/code/dataset/genki"

pain_path = os.path.join(ck_path, '11')
nopain_path = os.path.join(ck_path, '00')


# # Creat the list to store the data and label information
data_x = []
data_y = []
sum_nopain = 0
sum_pain = 0
datapath = os.path.join('data','GENKI_data.h5')
if not os.path.exists(os.path.dirname(datapath)):
    os.makedirs(os.path.dirname(datapath))

# order the file, so the training set will not contain the test set (don't random)
files = os.listdir(nopain_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(nopain_path,filename))
    sum_nopain = sum_nopain + 1
    data_x.append(I.tolist())
    data_y.append(0)


files = os.listdir(pain_path)
files.sort()
for filename in files:
    I = skimage.io.imread(os.path.join(pain_path,filename))
    sum_pain = sum_pain+1
    print(sum_pain)
    data_x.append(I.tolist())
    data_y.append(1)

print(np.shape(data_x))
print(np.shape(data_y))

datafile = h5py.File(datapath, 'w')
datafile.create_dataset("data_pixel", dtype = 'uint8', data=data_x)
datafile.create_dataset("data_label", dtype = 'int64', data=data_y)
datafile.close()

print("Save data finish!!!")
