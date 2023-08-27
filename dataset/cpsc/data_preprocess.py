# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-10 17:38

import numpy as np
import pandas as pd
import os
import scipy.io
from scipy.signal import resample

data_path = "."
def preprocess_physionet(data_path):
    """
    before running this data preparing code,
    please first download the raw data from https://physionet.org/content/challenge-2017/1.0.0/,
    and put it in data_path
    """

    # read label
    label_df = pd.read_csv(os.path.join(data_path, 'REFERENCE.csv'))
    label = label_df.iloc[:, 1].values
    # print(Counter(label))

    # read data
    all_data = []
    filenames = label_df.iloc[:, 0].values
    # filenames = filenames[0:100]
    print(len(filenames))
    label_ls = []
    for i, filename in enumerate(filenames):
        mat = scipy.io.loadmat(os.path.join(data_path, '{0}.mat'.format(filename)))
        mat = np.array(mat['ECG'][0][0][2]) # (12,7500)
        # print(mat)
        # print(mat.shape)
        # mat = mat[1, :]
        mat = resample(mat, int(mat.shape[1] / 5), axis=1) # (12,7500)
        mat = mat.transpose((1, 0))
        # print(mat.shape)
        mat1 = np.zeros((1000, 12))
        if mat.shape[0] < 1000:
            start = (1000 - mat.shape[0]) // 2
            mat1[start:start+mat.shape[0],:] = mat
        else:
            mat1 = mat[:1000, :]
        if label[i] == 1:   # Normal
            label_ls.append([1, 0, 0, 0, 0, 0, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 2:   # Normal
            label_ls.append([0, 1, 0, 0, 0, 0, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 3:   # Normal
            label_ls.append([0, 0, 1, 0, 0, 0, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 4:  # Normal
            label_ls.append([0, 0, 0, 1, 0, 0, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 5:  # Normal
            label_ls.append([0, 0, 0, 0, 1, 0, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 6:  # Normal
            label_ls.append([0, 0, 0, 0, 0, 1, 0, 0, 0])
            all_data.append(mat1)
        if label[i] == 7:  # Normal
            label_ls.append([0, 0, 0, 0, 0, 0, 1, 0, 0])
            all_data.append(mat1)
        if label[i] == 8:  # Normal
            label_ls.append([0, 0, 0, 0, 0, 0, 0, 1, 0])
            all_data.append(mat1)
        if label[i] == 9:  # Normal
            label_ls.append([0, 0, 0, 0, 0, 0, 0, 0, 1])
            all_data.append(mat1)
    all_data = np.array(all_data)
    label_ls = np.array(label_ls)
    np.save("test_ICBEB_signal", all_data)
    np.save("test_ICBEB_label_9", label_ls)


if __name__ == '__main__':
    preprocess_physionet(data_path)