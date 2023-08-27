
import json
import pandas as pd
import json

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
def ECGplot_multiLeads(data, fig_name, new_sig=None,save= False, i_path = None):
    '''绘制多导联'''
    # print(data)
    # data = np.random.rand(12, 5000)
    fig, ax = plt.subplots(figsize=(10, 9), dpi=200)
    ax.set_xticks(np.arange(0, 10.5, 0.2))
    ax.set_yticks(np.arange(-23, +1.0, 0.5))
    ax.tick_params(axis='both', which='both', bottom=False, top=False, left=False, right=False)
     # 隐藏 x 和 y 轴刻度标签的数字
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    #   # 设置 x 轴次要刻度线
    ax.minorticks_on()
    x_major_locator = MultipleLocator(0.04)  # 设置 x 轴主要刻度线每隔 1 个单位显示一个
    ax.xaxis.set_minor_locator(x_major_locator)
    ax.grid(which='major', linestyle='-', linewidth='0.3', color='gray')
    ax.grid(which='minor', linestyle='-', linewidth='0.1', color=(1, 0.7, 0.7))
    # ax.xaxis.set_major_locator(plt.MultipleLocator(0.2))
    # ax.yaxis.set_major_locator(plt.MultipleLocator(2))


    t = np.arange(0, len(data[0]) * 1 / 100, 1 / 100)
    lead = ["I","II","III","aVR","aVL","aVF","V1","V2","V3","V4","V5","V6"]
    # lead = ["I","II","III","aVR","aVL","aVF","V1","V2"]
    for i, l in enumerate(lead):
        ax.plot(t, np.array(data[i]) -2*i, label=l,linewidth=0.8, color='black')
    
    # ymin = -1.5
    # ymax = 1.5
    ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
    ax.set(xlabel='time (s)')
#     ax.set(ylabel='Voltage (mV)')
#     ax.autoscale(tight=True)
    ax.set_xlim(left=0, right=10.5)

    ax.set_ylim(top=1, bottom=-23)
    plt.savefig("./fig/" + fig_name + '.png')
    plt.show()


def ECGplot_multiLeads_mark(data,result):
    '''多导联＋标志点函数'''
    ## 原信号
    fig, ax = plt.subplots(figsize=(14, 8), dpi=200)
    ax.set_xticks(np.arange(0, 20.5, 1))
    ax.set_yticks(np.arange(-10.0, +2.0, 0.5))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.3', color=(1, 0.7, 0.7))

    ## 标志点
    otherlist=result['otherleadslist']
    mark_data = json.loads(otherlist)  # 解析 字符串为一个字典
    print(mark_data)
    t = np.arange(0, len(data['I']) * 1 / 500, 1 / 500)

    lead = ["I","II","III","aVR","aVL","aVF","V1","V5"]

    for i, l in enumerate(lead):
        # sig = np.array(data[l])/10000 # 心狗设备
        sig = np.array(data[l])*0.000000113  # 医院设备

    # --------if-else: 控制是否按照II导联标志点绘图----start-------
        # if  l=='I' or l=='II' or l=='III' or l=='aVR' or l=='aVL' or l=='aVF':
        #     mark = json.loads(mark_data['II'])  # 解析 字符串为一个字典
        # else:
    # --------if-else: 控制是否按照II导联标志点绘图-----end------
        mark = json.loads(mark_data[l])  # 解析 字符串为一个字典
        ax.plot(t, sig -2*i, label=l,color="k",linewidth=1)
        P1list = [int(i)for i in  mark["p1list"].split(",")[:-1]]
        P2list = [int(i) for i in  mark["p2list"].split(",")[:-1]]
        Plist = [int(i) for i in  mark["plist"].split(",")[:-1]]
        Qlist = [int(i) for i in  mark["qlist"].split(",")[:-1]]
        Q1list = [int(i) for i in  mark["q1list"].split(",")[:-1]]
        Rlist = [int(i) for i in  mark["rlist"].split(",")[:-1]]
        Slist = [int(i) for i in  mark["slist"].split(",")[:-1]]
        S2list = [int(i) for i in  mark["s2list"].split(",")[:-1]]
        Tlist = [int(i) for i in  mark["tlist"].split(",")[:-1]]
        T1list = [int(i) for i in  mark["t1list"].split(",")[:-1]]
        T2list = [int(i) for i in  mark["t2list"].split(",")[:-1]]

        if i==7:
            ax.scatter(np.array(P1list[:])/500, sig[P1list[:]]-2*i, s = 10,color="pink", label='P1')
            ax.scatter(np.array(Plist[:])/500, sig[Plist[:]]-2*i, s = 10,color="blueviolet",label='P')
            ax.scatter(np.array(P2list[:])/500, sig[P2list[:]]-2*i, s = 10,color="plum", label='P2')
            ax.scatter(np.array(Q1list[:])/500, sig[Q1list[:]]-2*i, s = 10,color="lime", label='Q1')
            ax.scatter(np.array(Qlist[:])/500, sig[Qlist[:]]-2*i, s = 10,color="limegreen",label='Q')
            ax.scatter(np.array(Rlist[:])/500, sig[Rlist[:]]-2*i, s = 10,color="red",label='R')
            ax.scatter(np.array(Slist[:])/500, sig[Slist[:]]-2*i, s = 10,color="dodgerblue", label='S')
            ax.scatter(np.array(S2list[:])/500, sig[S2list[:]]-2*i, s = 10,color="cyan", label='S2')
            ax.scatter(np.array(T1list[:])/500, sig[T1list[:]]-2*i, s = 10,color="yellow", label='T1')
            ax.scatter(np.array(Tlist[:])/500, sig[Tlist[:]]-2*i, s = 10,color="coral",label='T')
            ax.scatter(np.array(T2list[:])/500, sig[T2list[:]]-2*i, s = 10,color="gold", label='T2')

        else:
            ax.scatter(np.array(P1list[:])/500, sig[P1list[:]]-2*i, s = 10,color="pink")
            ax.scatter(np.array(Plist[:])/500, sig[Plist[:]]-2*i, s = 10,color="blueviolet")
            ax.scatter(np.array(P2list[:])/500, sig[P2list[:]]-2*i, s = 10,color="plum")
            ax.scatter(np.array(Q1list[:])/500, sig[Q1list[:]]-2*i, s = 10,color="lime")
            ax.scatter(np.array(Qlist[:])/500, sig[Qlist[:]]-2*i, s = 10,color="limegreen")
            ax.scatter(np.array(Rlist[:])/500, sig[Rlist[:]]-2*i, s = 10,color="red")
            ax.scatter(np.array(Slist[:])/500, sig[Slist[:]]-2*i, s = 10,color="dodgerblue")
            ax.scatter(np.array(S2list[:])/500, sig[S2list[:]]-2*i, s = 10,color="cyan")
            ax.scatter(np.array(T1list[:])/500, sig[T1list[:]]-2*i, s = 10,color="yellow")
            ax.scatter(np.array(Tlist[:])/500, sig[Tlist[:]]-2*i, s = 10,color="coral")
            ax.scatter(np.array(T2list[:])/500, sig[T2list[:]]-2*i, s = 10,color="gold")
        ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0))
        ax.set(xlabel='time(s)')
        ax.set(ylabel='Voltage (mV)')
        ax.autoscale(tight=True)
        ax.set_xlim(left=0, right=16)
        # ax.set_ylim(top=ymax, bottom=ymin)
    plt.show()

import pickle
def load_clinical_label_and_report():

    label_list = {'I°房室传导阻滞':"first degree AV block:focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.",
                  'T波倒置':"inverted T-waves",
                  "不齐":"Sinus arrhythmia:Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
                  "完全性右束支传导阻滞":"complete right bundle branch block:Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in these leads. Additionally, the presence of a secondary R wave in lead V1 may indicate RBBB.",
                  '完全性左束支传导阻滞':"complete left bundle branch block",
                  "室上性心动过速":"Supraventricular Tachycardia:Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
                  "室性早搏":"ventricular premature complex:Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
                  "左前分支传导阻滞":"Left anterior fascicular block:Look for left axis deviation (> -45 degrees) and qR pattern in lead I, and rS pattern in lead aVL.",
                  "心房扑动":"Atrial Flutter:Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
                  "心房颤动":"Atrial Fibrillation:Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
                  "房性早搏":"atrial premature complex:Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
                  "正常心电图":"Normal ECG",
                  "窦性心动过缓":"Sinus Bradycardia",
                  "窦性心动过速":"Sinus Tachycardia"}
    # label_list = {
    #     'I°房室传导阻滞': "first degree AV block",
    #     'T波倒置': "inverted T-waves",
    #     "不齐": "Sinus arrhythmia",
    #     "完全性右束支传导阻滞": "complete right bundle branch block",
    #     '完全性左束支传导阻滞': "complete left bundle branch block",
    #     "室上性心动过速": "Supraventricular Tachycardia",
    #     # "室性早搏": "ventricular premature complex:Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
    #     "室性早搏": "ventricular premature complex",
    #     "左前分支传导阻滞": "Left anterior fascicular block",
    #     "心房扑动": "Atrial Flutter",
    #     "心房颤动": "Atrial Fibrillation",
    #     # "房性早搏": "atrial premature complex:Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
    #     "房性早搏": "atrial premature complex",
    #     "正常心电图": "Normal Electrocardiogram",
    #     "窦性心动过缓": "Sinus Bradycardia",
    #     "窦性心动过速": "Sinus Tachycardia"}
    f = open('/home/tyy/unECG/dataset/clinical_dataset/mlb12.pkl', 'rb')
    data = pickle.load(f)
    return [label_list[item] for item in data.classes_]

def get_label(X_data, Y_data):

    for idx in range(len(X_data)):
        label_list = Y_data[idx]
        disease_label_index = np.where(label_list == 1)[0]
        background_list, label_list = [], []
        for sub_idx in disease_label_index:
            sub_label = label_name[sub_idx]
            label_list.append(sub_label)
        background_info = ". ".join(background_list)
        diagnosis = ", ".join(label_list)

from unECG.plot_visualization import get_ecg_feature_and_label
if __name__ == '__main__':
    # X = np.load("../dataset/clinical_dataset/X_clinical_data_12.npy", allow_pickle=True)
    # y = np.load("../dataset/clinical_dataset/y_clinical_data_12.npy", allow_pickle=True)
    X = np.load("../dataset/georgia/signal_data_filter_100.npy", allow_pickle=True)
    y = np.load("../dataset/georgia/label_data_filter_100.npy", allow_pickle=True)
    # X_data, y_data = X[0:10], y[0:10]
    label_name = load_clinical_label_and_report()
    signal, report, label_index, text_list, random_index = get_ecg_feature_and_label(X, y, "Incomplete right bundle branch block")
    ECGplot_multiLeads(signal.T, "signal_" + str(label_index))
    # for idx in range(10):
    #     label_list = y_data[idx]
    #     disease_label_index = np.where(label_list == 1)[0]
    #     background_list, label_list = [], []
    #     for sub_idx in disease_label_index:
    #         sub_label = label_name[sub_idx]
    #         label_list.append(sub_label)
    #     print("signal_" + str(idx) + "的label是："+ ", ".join(label_list))
    #     ECGplot_multiLeads(X_data[idx].T, "signal_" + str(idx))