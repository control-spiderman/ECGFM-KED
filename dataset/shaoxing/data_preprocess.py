# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-20 15:39

from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pandas as pd
import numpy as np
from scipy.signal import resample

def handler_data():
    """
        before running this data preparing code,
        please first download the raw data from https://doi.org/10.6084/m9.figshare.c.4560497.v2,
        and put it in data_path
        """
    mlb = MultiLabelBinarizer()
    """"""
    row_data_file = pd.read_excel('Diagnostics.xlsx')
    print(row_data_file.shape)
    print(row_data_file.head())

    # 先将所有的signal读取出来
    signal_data = []
    base_path = '/home/lzy/workspace/dataSet/shaoxing_hospital/ECGData/'
    error_file = []
    error_index = []
    for idx, item in row_data_file.iterrows():
        file_path = base_path + item['FileName'] + '.csv'
        file_data = pd.read_csv(file_path)
        return_result = _resample_signal(file_data.values)
        if return_result.sum() == 0:
            error_file.append(item['FileName'])
            error_index.append(idx)
        signal_data.append(return_result)
    signal_data = np.array(signal_data)
    signal_data = np.delete(signal_data, error_index, axis=0)
    # 处理label
    row_data_file = row_data_file.drop(error_index)
    counts = pd.Series(row_data_file.Rhythm.values).value_counts()
    counts = counts[counts > 100]
    row_data_file.Rhythm = row_data_file.Rhythm.apply(lambda x: list(set([x]).intersection(set(counts.index.values))))
    row_data_file['Rhythm_len'] = row_data_file.Rhythm.apply(lambda x: len(x))
    X = signal_data[row_data_file.Rhythm_len > 0]
    Y = row_data_file[row_data_file.Rhythm_len > 0]
    mlb.fit(Y.Rhythm.values)
    y = mlb.transform(Y.Rhythm.values)
    X = np.stack(X)
    X.dump("signal_data.npy")
    y.dump("label_data.npy")
    # save LabelBinarizer
    with open('mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)

def _resample_signal(mat):
    mat = resample(mat, int(mat.shape[0] / 5), axis=0)
    mat = mat / 1000
    mat1 = np.zeros((1000, 12))
    if mat.shape[0] == 0:
        return mat1
    if mat.shape[0] < 1000:
        start = (1000 - mat.shape[0]) // 2
        mat1[start:start + mat.shape[0], :] = mat
    else:
        mat1 = mat[:1000, :]
    return mat1

import pickle
if __name__ == '__main__':
    handler_data()
    # f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # print(data.classes_)

    # item_count = []
    # data = np.load('signal_data.npy', allow_pickle=True)
    # for item in data:
    #     if item.sum() == 0:
    #         item_count.append(item)
    # print(len(item_count))
    # print(data.shape)