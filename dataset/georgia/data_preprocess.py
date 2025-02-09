# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-14 22:26

import os
import csv
import numpy as np
from scipy.io import loadmat
import matplotlib.pylab as plt
import pandas as pd

file_path = "XXX/georgia/"  # Replace with your folder path
csv_file_path = "XXX/georgia/label_data.csv"  # Replace with your desired output csv file path
output_data_path = "XXX/georgia/signal_data.npy"  # Replace with your desired output data npy file path

def data_load():
    headers = ['filename', 'age', 'sex', 'Dx', 'Rx', 'Hx', 'Sx']  # Headers of csv file
    output_data = []
    folds = ["g1", "g2", "g3", "g4", "g5", "g6", "g7", "g8", "g9"]

    from scipy.signal import resample
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)  # Write headers to csv file
        error_list = []
        for fold in folds:
            folder_path = file_path +fold + "/"
            for filename in os.listdir(folder_path):
                if filename.endswith(".hea"):  # Only consider files with .hea extension
                    entry = [None] * len(headers)  # Initialize an empty list for each entry

                    # Open .hea file and read relevant lines
                    with open(os.path.join(folder_path, filename), 'r') as f:
                        lines = f.readlines()

                    # Extract filename and Dx
                    entry[0] = filename[:-4]  # Remove .hea extension from filename
                    # entry[3] = [d for d in lines[3].split(':')[1].split(',')] # Extract Dx and convert to list

                    # Extract age, sex, Rx, Hx, Sx if available
                    for line in lines:
                        if line.startswith('# Dx'):
                            entry[3] = line.split(': ')[1].replace("[","").replace("]","").replace("\n", "").split(',')
                        elif line.startswith('# Age'):
                            entry[1] = line.split(': ')[1].strip()
                        elif line.startswith('# Sex'):
                            entry[2] = line.split(': ')[1].strip()
                        elif line.startswith('# Rx'):
                            entry[4] = line.split(': ')[1].strip()
                        elif line.startswith('# Hx'):
                            entry[5] = line.split(': ')[1].strip()
                        elif line.startswith('# Sx'):
                            entry[6] = line.split(': ')[1].strip()

                    writer.writerow(entry)  # Write entry to csv file

                    # Read corresponding .mat file
                    mat_filename = os.path.join(folder_path, entry[0] + '.mat')

                    if os.path.exists(mat_filename):
                        data = loadmat(mat_filename)['val']
                        data = data / 1000
                        mat = resample(data, int(data.shape[1] / 5), axis=1)  # (12,7500)
                        mat = mat.transpose((1, 0))
                        # print(mat.shape)
                        mat1 = np.zeros((1000, 12))
                        if mat.shape[0] < 1000:
                            start = (1000 - mat.shape[0]) // 2
                            mat1[start:start + mat.shape[0], :] = mat
                        else:
                            mat1 = mat[:1000, :]
                        output_data.append(mat1)
                    else:
                        error_list.append(mat_filename)
                        print("error file:", mat_filename)

    # Save data as numpy array
    output_data_array = np.array(output_data)
    print(output_data_array)
    # np.save(output_data_path, output_data_array)

dx_map_path = "/home/tyy/unECG/dataset/Dx_map.csv"
def get_disease_label():
    dx_abbrs_set = set()
    # 读取第二个CSV文件为字典，包含SNOMED CT Code和Abbreviation两列
    dx_dict = {}
    with open(dx_map_path, newline='') as file:
        reader = csv.DictReader(file)
        for row in reader:
            dx_dict[row['SNOMED CT Code']] = [row['Abbreviation'], row['Dx']]

    # 遍历第一个CSV文件
    with open(csv_file_path, newline='') as file:
        reader = csv.reader(file)
        next(reader)  # 跳过表头
        rows = []
        for row in reader:
            # 获取第四列的list值
            dx_codes = eval(row[3])
            dx_names = []
            dx_abbrs = []
            # 遍历list，查找每个编码对应的Dx文本和Abbreviation
            for code in dx_codes:
                if code in dx_dict:
                    dx_abbrs.append(dx_dict[code][0])
                    dx_names.append(dx_dict[code][1])
                    dx_abbrs_set.add(dx_dict[code][0])
                else:
                    dx_abbrs.append('')
                    dx_names.append('')
            # 将查找到的Dx文本和Abbreviation添加到原来的行末尾
            row.extend([dx_names, dx_abbrs])
            rows.append(row)
    # dx_abbrs_list = list(dx_abbrs_set)
    # for row in rows:

    # 将修改后的文件写出
    with open('/home/tyy/unECG/dataset/georgia/final_label_data.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['filename', 'age', 'sex', 'Dx', 'Rx', 'Hx', 'Sx', 'Dx Names', 'Dx Abbreviations'])
        writer.writerows(rows)

LABEL_LIST = ["1st degree AV block",
            "Atrial fibrillation",  # 差
            "Atrial flutter",
            "Incomplete right bundle branch block",
            "Left anterior fascicular block",
            "Left axis deviation",
            "Left bundle branch block",
            "Low QRS voltages", # 差
            "Nonspecific intraventricular conduction disorder",
            "Premature atrial contraction", # 差
            "Prolonged QT interval",
            "Q wave abnormal",  # 差
            "Right bundle branch block",
            "Sinus arrhythmia", # 差
            "Sinus bradycardia",    # 差
            "Sinus rhythm", # 差
            "Sinus tachycardia",
            "T wave abnormal",
            "T wave inversion", # 差
            "Ventricular premature beats"
            ]
label_abbr_list = [
    "IAVB", # 0.34
    "AF",
    "AFL",
    "IRBBB",
    "LAnFB",
    "LAD",
    "LBBB",
    "LQRSV",    # 0.58
    "NSIVCB",
    "PAC",
    "LQT",  # 0.234
    "QAb",  # 0.07
    "RBBB", # 0.41
    "SA",
    "SB",   # 0.05
    "NSR",  # 0.17
    "STach",
    "TAb",
    "TInv", # 0.20
    "VPB"
]
import ast
from sklearn.preprocessing import MultiLabelBinarizer
import pickle
def add_one_hot_label():
    mlb = MultiLabelBinarizer()
    signal_data = np.load('signal_data.npy')
    label_file = pd.read_csv('final_label_data.csv')
    label_file['Dx Names'] = label_file['Dx Names'].apply(lambda x: ast.literal_eval(x))
    label_file['Dx Abbreviations'] = label_file['Dx Abbreviations'].apply(lambda x: ast.literal_eval(x))

    counts = pd.Series(np.concatenate(label_file['Dx Abbreviations'].values)).value_counts()
    counts = counts[counts > 100]
    counts = counts.loc[label_abbr_list]
    label_file['Dx Abbreviations'] = label_file['Dx Abbreviations'].apply(lambda x: list(set(x).intersection(set(counts.index.values))))
    label_file['Dx_Abbr_len'] = label_file['Dx Abbreviations'].apply(lambda x: len(x))
    X = signal_data[label_file.Dx_Abbr_len > 0]
    Y = label_file[label_file.Dx_Abbr_len > 0]
    mlb.fit(Y['Dx Abbreviations'].values)
    y = mlb.transform(Y['Dx Abbreviations'].values) # (8958, 29)
    with open('mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)
    # 存储X， y
    np.save('signal_data_filter_100.npy', X)
    np.save('label_data_filter_100.npy', y)


import json
if __name__ == '__main__':
    data_load()
    # add_one_hot_label()

