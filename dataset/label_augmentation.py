# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-09 16:53

import numpy as np
import pandas as pd
import aiohttp
import requests


def preprocess(X, y):
    new_X = []
    new_y = []
    error_list = []
    for i in range(len(y)):
        if ('Ö' in y[i,0]) or ('Å' in y[i, 0]) or ('ö' in y[i, 0]) or ('ä' in y[i, 0]):
            error_list.append(i)
            pass
        elif ('4.46' in y[i, 0]):
            split_text = y[i, 0].split(' ')
            idx = split_text.index('4.46')
            split_text = split_text[:idx]
            joined_text = ' '.join(split_text)
            new_X.append(X[i, :].tolist())
            replaceed = [joined_text, y[i, 1]]
            new_y.append(replaceed)
        else:
            same = [y[i, 0], y[i, 1]]
            new_y.append(same)
            new_X.append(X[i, :].tolist())
    print(error_list)
    new_array_X = np.array(new_X)
    new_array_y = np.array(new_y)
    return new_array_X, new_array_y

def data_reprocess():
    X_test = np.load('../X_all.npy')    # ecg信号
    y_test = np.load('../y_all.npy', allow_pickle=True)     # label，每个样本都是一个list，存在多标签情况
    y_report = np.load('../y_report_all_trans.npy', allow_pickle=True)      # report
    y = np.concatenate((y_report.reshape((-1, 1)), y_test.reshape((-1, 1))), axis=1)
    print(y.shape)
    print(y[:5,:])
    new_X, new_y = preprocess(X_test, y)
    print(len(new_X))
    print(len(new_y))
    np.save("signal_filtered", new_X)
    np.save("labelReport_filtered", new_y)

import json
import time
def generate_all_label_augment(use_what_label=None):
    scp_statement = pd.read_csv('scp_statements.csv', index_col=[0])
    # 服了先使用all label吧
    # if use_what_label == "diagnostic":
    #     label_data = scp_statement.loc[scp_statement["diagnostic"] == 1]["description"]
    # elif use_what_label == "form":
    #     label_data = scp_statement.loc[scp_statement["form"] == 1]["description"]
    # elif use_what_label == "rhythm":
    #     label_data = scp_statement.loc[scp_statement["rhythm"] == 1]["description"]
    # elif use_what_label == "all":
    #     all_label = scp_statement["description"].values
    # elif use_what_label == "subdiagnostic":
    #     label_data = scp_statement.loc[scp_statement["diagnostic"] == 1]["diagnostic_subclass"]
    # elif use_what_label == "superdiagnostic":
    #     label_data = scp_statement.loc[scp_statement["diagnostic"] == 1]["diagnostic_class"]
    # print(len(label_data))
    all_label = scp_statement["description"]
    all_label_map = {}
    for item in all_label.items():
        print(item)
        all_label_map[item[0]] = item[1]
    with open("all_label_map.json", "w") as f:
        json.dump(all_label_map, f)
    _handler_generate_augment(scp_statement, use_what_label)



def _handler_generate_augment(scp_statement, use_what_label):
    diagnosis_label = scp_statement.loc[scp_statement["diagnostic"] == 1]["description"]
    form_label = scp_statement.loc[scp_statement["form"] == 1]["description"]
    rhythm_label = scp_statement.loc[scp_statement["rhythm"] == 1]["description"]
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                              "to diagnose abnormal electrocardiograms. How to diagnose"
    prompt_suffix_diagnosis = "from 12-lead ECG, such as what leads or what features to focus on ,etc. Your answer must be less " \
                              "than 50 words."
    prompt_prefix_form = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                         "to identify abnormal electrocardiograms. How to identify"
    prompt_suffix_form = "from 12-lead ECG, such as what leads or what features to focus on ,etc. Your answer must be less " \
                         "than 100 words."
    prompt_prefix = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                    "to identify abnormal electrocardiograms. How to identify"
    prompt_suffix = "from 12-lead ECG, such as what leads or what features to focus on ,etc. Your answer must be less " \
                    "than 100 words."
    url = "CHAT_WITH_YOUR_GPT"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    label_argment = {}
    error_list = []
    for item in diagnosis_label:
        try:
            data = {"messages": [{"role": "user", "content": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis}],
                    "userId": "serveForPaper"}
            json_data = json.dumps(data)
            response = requests.post(url=url, data=json_data, headers=headers)
            json_response = response.json()
            print(json_response["content"])
            label_argment[item] = json_response["content"].replace("\n\t", "").replace("\n", "")
        except Exception as e:
            error_list.append(item)
            print(e)
    for item in form_label:
        try:
            data = {"messages": [{"role": "user", "content": prompt_prefix_form + item + prompt_suffix_form}],
                    "userId": "serveForPaper"}
            json_data = json.dumps(data)
            response = requests.post(url=url, data=json_data, headers=headers)
            json_response = response.json()
            print(json_response["content"])
            label_argment[item] = json_response["content"].replace("\n\t", "").replace("\n", "")
        except Exception as e:
            error_list.append(item)
            print(e)
    for item in rhythm_label:
        try:
            data = {"messages": [{"role": "user", "content": prompt_prefix + item + prompt_suffix}],
                    "userId": "serveForPaper"}
            json_data = json.dumps(data)
            response = requests.post(url=url, data=json_data, headers=headers)
            json_response = response.json()
            print(json_response["content"])
            label_argment[item] = json_response["content"].replace("\n\t", "").replace("\n", "")
        except Exception as e:
            error_list.append(item)
            print(e)
    print(error_list)
    with open("all_label_augment.json", "w") as f:
        json.dump(label_argment, f)
    print(label_argment)

# def _handler_generate_augment_(right_label, error_label):
#     prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
#                               "to diagnose abnormal electrocardiograms. How to avoid diagnosing"
#
#     prompt_suffix_diagnosis = ". Your answer must be less " \
#                               "than 20 words."
#     url = "CHAT_WITH_YOUR_GPT"
#     headers = {"Content-Type": "application/json;charset=utf-8",
#                "Accept": "*/*",
#                "Accept-Encoding": "gzip, deflate, br",
#                "Connection": "keep-alive"}
#     data = {"messages": [{"role": "user", "content": prompt_prefix_diagnosis + right_label + " as "+ error_label + prompt_suffix_diagnosis}],
#             "userId": "serveForPaper"}
#     json_data = json.dumps(data)
#     response = requests.post(url=url, data=json_data, headers=headers)
#     json_response = response.json()
#     print(json_response["content"])

def _handler_generate_augment_(item, prompt_prefix=None, prompt_suffix=None):

    if prompt_prefix:
        prompt_prefix_diagnosis = prompt_prefix
    else:
        prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                                  "to diagnose "
    if prompt_suffix:
        prompt_suffix_diagnosis = prompt_suffix
    else:
        prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                              "than 50 words."
    url = "CHAT_WITH_YOUR_GPT"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    data = {"messages": [{"role": "user", "content": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis}],
            "userId": "serveForPaper"}
    json_data = json.dumps(data)
    response = requests.post(url=url, data=json_data, headers=headers)
    json_response = response.json()
    print(json_response["content"])
    return json_response["content"]

def handler_23_cla_augment():
    origin_label_map = {'NORM':"normal ECG",
                                'IMI':"inferior myocardial infarction",
                                'AMI':"anterior myocardial infarction",
                                'STTC':"ST/T-changes",
                                'LVH':"left ventricular hypertrophy",
                                'LAFB/LPFB':"left anterior/posterior fascicular block",
                                'ISC_':"non-specific ischemic",
                                'IRBBB':"incomplete right bundle branch block",
                                'ISCA':"ischemic in anterolateral/anteroseptal/anterior leads",
                                '_AVB':"AV block",
                                'IVCD':"non-specific intraventricular conduction disturbance (block)",
                                'NST_':"non-specific ST changes",
                                'CRBBB':"complete right bundle branch block",
                                'CLBBB':"complete left bundle branch block",
                                'LAO/LAE':"left atrial overload/enlargement",
                                'ISCI':"ischemic in inferior/inferolateral leads",
                                'LMI':"lateral myocardial infarction",
                                'RVH':"right ventricular hypertrophy",
                                'RAO/RAE':"right atrial overload/enlargement",
                                'WPW':"Wolf-Parkinson-White syndrome",
                                'ILBBB':"incomplete left bundle branch block",
                                'SEHYP':"septal hypertrophy",
                                'PMI':"posterior myocardial infarction"}
    label_augment_23 = {}
    for key , value in origin_label_map.items():
        augment_result = _handler_generate_augment_(value)
        label_augment_23[value] = augment_result
    with open("label_augment_23.json", "w") as f:
        json.dump(label_augment_23, f)

import pickle
def handler_finetune_dataset_augment():
    all_label_map = {"SB": "Sinus Bradycardia", "SR": "Sinus Rhythm", "AFIB": "Atrial Fibrillation",
                     "ST": "Sinus Tachycardia", "AF": "Atrial Flutter", "SA": "Sinus Arrhythmia",
                     "SVT": "Supraventricular Tachycardia", "AT": "Atrial Tachycardia"}
    f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
    data = pickle.load(f)
    shaoxing_label_name = [all_label_map[item] for item in data.classes_]

    with open("/home/tyy/unECG/dataset/georgia/label_map.json", 'r') as f:
        all_label_map = json.load(f)
    f = open('/home/tyy/unECG/dataset/georgia/mlb.pkl', 'rb')
    data = pickle.load(f)
    georgia_label_name = [all_label_map[item] for item in data.classes_]
    cpsc_label_name = ['normal ECG', 'Atrial fibrillation', 'first degree AV block', 'left bundle branch block',
                       'Right bundle branch block',
                       "atrial premature complex", "ventricular premature complex", 'non-specific ST depression',
                       'non-specific ST elevation']
    shaoxing_label_map, georgia_label_map, cpsc_label_map = {}, {}, {}
    for item in shaoxing_label_name:
        shaoxing_label_map[item] = _handler_generate_augment_(item)
    for item in georgia_label_name:
        georgia_label_map[item] = _handler_generate_augment_(item)
    for item in cpsc_label_name:
        cpsc_label_map[item] = _handler_generate_augment_(item)
    with open("cpsc_label_map_report.json", "w") as f:
        json.dump(cpsc_label_map, f)
    with open("./shaoxing/shaoxing_label_map_report.json", "w") as f:
        json.dump(shaoxing_label_map, f)
    with open("./georgia/georgia_label_map_report.json", "w") as f:
        json.dump(georgia_label_map, f)


def generate_background_info_by_prompt():
    with open("all_label_map.json", 'r') as f:
        all_label_map = json.load(f)
    prompt_prefix = "I am a cardiology intern. As a professional Electrocardiologist, could you please tell me how to diagnose "
    prompt_suffix = " from ECG in plain language. Your answer must be less than 50 words."
    label_map = {}
    for item in all_label_map.values():
        label_map[item] = _handler_generate_augment_(item, prompt_prefix=prompt_prefix, prompt_suffix=prompt_suffix)
    with open("label_map_intern.json", "w") as f:
        json.dump(label_map, f)

"""
label_map_concise版本：Please teach me how to diagnose XXX from 12-lead ECG. Your answer must be less than 50 words.
label_map_plain版本：Please explain XXX in ECG to me in plain language. Your answer must be less than 50 words.
label_map_plain_diagnosis版本：Please explain XXX in ECG to me in plain language and tell me how to identify it. Your answer must be less than 50 words.
label_map_intern: I am a cardiology intern. As a professional Electrocardiologist, could you please tell me how to diagnose XXX from ECG in plain language
"""

if __name__ == '__main__':
    """"""
    # generate_all_label_augment(
    # _handler_generate_augment([ "First-degree atrioventricular block","Left bundle branch block",
    #                        "Right bundle branch block", "ST-segment depression", "ST-segment elevated"])
    # _handler_generate_augment_("ST-segment elevated", "Normal ECG")

    # _handler_generate_augment_("I want you to play the role of a professional Electrocardiologist, and I need you to teach me how "
    #                            "to identify ST depression and ST elevated from a ST-T change ECG. Your answer must be less than 20 words.")

    # handler_finetune_dataset_augment()

    generate_background_info_by_prompt()