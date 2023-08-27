# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-07-24 12:48

import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn import manifold
from sklearn.manifold import TSNE
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def visual(feat):
    # t-SNE的最终结果的降维与可视化
    ts = manifold.TSNE(n_components=3, init='pca', random_state=0)

    x_ts = ts.fit_transform(feat)

    print(x_ts.shape)  # [num, 2]

    x_min, x_max = x_ts.min(0), x_ts.max(0)

    x_final = (x_ts - x_min) / (x_max - x_min)

    return x_final


# 设置散点形状
# maker = ['o', 's', '^', 's', 'p', '*', '<', '>', 'D', 'd', 'h', 'H']
maker = ["o", "s", "+", "x", "*", "D", "d", "v", "^", "<", ">", "p", "P", "h", "H", ".", ",", "1", "2", "3"]
# 设置散点颜色
colors = ['#e38c7a', '#656667', '#99a4bc', 'cyan', 'blue', 'lime', 'r', 'violet', 'm', 'peru', 'olivedrab',
          'hotpink']
# 图例名称
Label_Com = ['a', 'b', 'c', 'd']
# 设置字体格式
font1 = {'family': 'Times New Roman',
         'weight': 'bold',
         'size': 32,
         }

def plotlabels3D(S_lowDWeights, Trure_labels, name, label_nums=9):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'z': S_data[:, 2], 'label': S_data[:, 3]})
    print(S_data)
    print(S_data.shape)  # [num, 4]

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')

    for index in range(9):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        Z = S_data.loc[S_data['label'] == index]['z']
        ax.scatter(X, Y, Z, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

    ax.set_xticks([])  # 去掉横坐标值
    ax.set_yticks([])  # 去掉纵坐标值
    ax.set_zticks([])  # 去掉深度坐标值

    ax.set_title(name, fontsize=32, fontweight='normal', pad=20)

def plotlabels(S_lowDWeights, Trure_labels, name, label_nums=9):
    True_labels = Trure_labels.reshape((-1, 1))
    S_data = np.hstack((S_lowDWeights, True_labels))  # 将降维后的特征与相应的标签拼接在一起
    S_data = pd.DataFrame({'x': S_data[:, 0], 'y': S_data[:, 1], 'label': S_data[:, 2]})
    print(S_data)
    print(S_data.shape)  # [num, 3]

    for index in range(9):  # 假设总共有三个类别，类别的表示为0,1,2
        X = S_data.loc[S_data['label'] == index]['x']
        Y = S_data.loc[S_data['label'] == index]['y']
        plt.scatter(X, Y, cmap='brg', s=100, marker=maker[index], c=colors[index], edgecolors=colors[index], alpha=0.65)

        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])  # 去掉纵坐标值

    plt.title(name, fontsize=32, fontweight='normal', pad=20)

def visualization_tsne(ecg_features, label):
    """"""
    label_nums = label.shape[1]
    selected_classes = [1,4,6]  # 排除1,16,
    # selected_classes = [1,4,7]  # 排除1,16,
    # selected_classes = [9,10,12,14,19]  # 排除1,16,
    # selected_features = ecg_features.cpu().numpy()[label.cpu().numpy()[:, selected_labels].any(dim=1)]
    # t-SNE降维
    tsne = TSNE(n_components=2, random_state=42)
    features_tsne = tsne.fit_transform(ecg_features.cpu().numpy())
    label_list = ["Atrial Fibrillation", "Sinus Bradycardia", "Sinus Rhythm"]
    plt.figure(figsize=(10, 10))
    for idx, i in enumerate(selected_classes):
    # for i in range(label_nums):
        if i >= 20:
            continue
        class_indices = np.argwhere(label.cpu().numpy()[:, i] == 1).flatten()
        # plt.scatter(features_tsne[class_indices, 0], features_tsne[class_indices, 1],
        #             label=f'class {i}', marker=maker[i])
        plt.scatter(features_tsne[class_indices, 0], features_tsne[class_indices, 1],cmap='brg', s=100,
                label=label_list[idx], marker=maker[i], alpha=0.65)
        plt.xticks([])  # 去掉横坐标值
        plt.yticks([])
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Atrial Fibrillation, Sinus Bradycardia, Sinus Rhythm
    feat = torch.rand(128, 1024)  # 128个特征，每个特征的维度为1024
    label_test1 = [0 for index in range(40)]
    label_test2 = [1 for index in range(40)]
    label_test3 = [2 for index in range(48)]

    label_test = np.array(label_test1 + label_test2 + label_test3)
    print(label_test)
    print(label_test.shape)

    fig = plt.figure(figsize=(10, 10))

    plotlabels(visual(feat), label_test, '(a)')

    plt.show()