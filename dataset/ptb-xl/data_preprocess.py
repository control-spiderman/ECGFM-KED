import pandas as pd
import numpy as np
import multiprocessing
from itertools import repeat
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import ast
import pickle
import csv
import json
import requests
import os
import wfdb
from tqdm import tqdm

"""
This data processing document has been modified from this project: https://github.com/helme/ecg_ptbxl_benchmarking
"""

def handler_data(experiment_name, task, datafolder, sampling_frequency = 100, min_samples = 0, train_fold = 8, val_fold = 9, test_fold = 10, folds_type = 'strat'):
    data, raw_labels = load_dataset(datafolder, sampling_frequency)

    # Preprocess label data：(21799,29)
    labels = compute_label_aggregations(raw_labels, datafolder, task)

    # Select relevant data and convert to one-hot (21799,1000,12), (21799,30)   Y:(21388, 71)
    data, labels, Y, _ = select_data(data, labels, task, min_samples,
                                                          './' + experiment_name + '/data/')
    input_shape = data[0].shape

    # 10th fold for testing (9th for now)   (2158,5)
    X_test = data[labels.strat_fold == test_fold]
    y_test = Y[labels.strat_fold == test_fold]
    # 9th fold for validation (8th for now) (2146,5)
    X_val = data[labels.strat_fold == val_fold]
    y_val = Y[labels.strat_fold == val_fold]
    # rest for training (17084,5)
    X_train = data[labels.strat_fold <= train_fold]
    y_train = Y[labels.strat_fold <= train_fold]

    # Preprocess signal data
    X_train, X_val, X_test = preprocess_signals(X_train, X_val, X_test,
                                                                     './' + experiment_name + '/data/')
    n_classes = y_train.shape[1]

    # 10th fold for testing (9th for now)   (2158,5)
    report_test = labels.report[labels.strat_fold == test_fold]
    # 9th fold for validation (8th for now) (2146,5)
    report_val = labels.report[labels.strat_fold == val_fold]
    # rest for training (17084,5)
    report_train = labels.report[labels.strat_fold <= train_fold]

    report_train.to_csv('./' + experiment_name+"/data/report_index_temp.csv", index=True)

    # utils.translate_report(report_train.values, './' + experiment_name + '/data/report_train.csv')
    # report_train.dump('./' + experiment_name + '/data/report_train.npy')

    # report_val = utils.translate_report(report_val.values,'./' + experiment_name + '/data/report_val.csv')
    # report_val.dump('./' + experiment_name + '/data/report_val.npy')
    #
    # report_test = utils.translate_report(report_test.values)
    # report_test.dump('./' + experiment_name+ '/data/report_test.npy')

    # save train and test labels
    # labels.dump('./' + experiment_name+ '/data/X_train.npy')
    X_train.dump('./' + experiment_name + '/data/X_train.npy')
    X_val.dump('./' + experiment_name + '/data/X_val.npy')
    X_test.dump('./' + experiment_name + '/data/X_test.npy')
    y_train.dump('./' + experiment_name + '/data/y_train.npy')
    y_val.dump('./' + experiment_name + '/data/y_val.npy')
    y_test.dump('./' + experiment_name + '/data/y_test.npy')

def load_raw_data_ptbxl(df, sampling_rate, path):
    if sampling_rate == 100:
        if os.path.exists(path + 'raw100.npy'):
            data = np.load(path+'raw100.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_lr)]
            data = np.array([signal for signal, meta in data])      # (21799, 1000, 12)
            pickle.dump(data, open(path+'raw100.npy', 'wb'), protocol=4)
    elif sampling_rate == 500:
        if os.path.exists(path + 'raw500.npy'):
            data = np.load(path+'raw500.npy', allow_pickle=True)
        else:
            data = [wfdb.rdsamp(path+f) for f in tqdm(df.filename_hr)]
            data = np.array([signal for signal, meta in data])
            pickle.dump(data, open(path+'raw500.npy', 'wb'), protocol=4)
    return data
def load_dataset(path, sampling_rate, release=False):
    if path.split('/')[-2] == 'ptbxl':
        # load and convert annotation data
        Y = pd.read_csv(path+'ptbxl_database.csv', index_col='ecg_id')  # (21799, 27)
        Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))

        # Load raw signal data
        X = load_raw_data_ptbxl(Y, sampling_rate, path)

    # elif path.split('/')[-2] == 'ICBEB':
    #     # load and convert annotation data
    #     Y = pd.read_csv(path+'icbeb_database.csv', index_col='ecg_id')
    #     Y.scp_codes = Y.scp_codes.apply(lambda x: ast.literal_eval(x))
    #
    #     # Load raw signal data
    #     X = load_raw_data_icbeb(Y, sampling_rate, path)

    return X, Y

def select_data(XX,YY, ctype, min_samples, outputfolder):
    # convert multilabel to multi-hot
    mlb = MultiLabelBinarizer()

    if ctype == 'diagnostic':
        X = XX[YY.diagnostic_len > 0]
        Y = YY[YY.diagnostic_len > 0]
        mlb.fit(Y.diagnostic.values)
        y = mlb.transform(Y.diagnostic.values)
        # 筛选出diagnostic_len为0的index和report
        # res_X = XX[YY.diagnostic_len == 0]
        # res_Y = YY[YY.diagnostic_len == 0]
        # res_YY = res_Y[YY.strat_fold <= 8]
        # res_YY.to_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/res_labels.csv", index=True)
    elif ctype == 'subdiagnostic':
        counts = pd.Series(np.concatenate(YY.subdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.subdiagnostic = YY.subdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['subdiagnostic_len'] = YY.subdiagnostic.apply(lambda x: len(x))
        X = XX[YY.subdiagnostic_len > 0]
        Y = YY[YY.subdiagnostic_len > 0]
        mlb.fit(Y.subdiagnostic.values)
        y = mlb.transform(Y.subdiagnostic.values)
    elif ctype == 'superdiagnostic':
        counts = pd.Series(np.concatenate(YY.superdiagnostic.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.superdiagnostic = YY.superdiagnostic.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['superdiagnostic_len'] = YY.superdiagnostic.apply(lambda x: len(x))
        X = XX[YY.superdiagnostic_len > 0]
        Y = YY[YY.superdiagnostic_len > 0]
        mlb.fit(Y.superdiagnostic.values)
        y = mlb.transform(Y.superdiagnostic.values)
    elif ctype == 'form':
        # filter
        counts = pd.Series(np.concatenate(YY.form.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.form = YY.form.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['form_len'] = YY.form.apply(lambda x: len(x))
        # select
        X = XX[YY.form_len > 0]
        Y = YY[YY.form_len > 0]
        mlb.fit(Y.form.values)
        y = mlb.transform(Y.form.values)
    elif ctype == 'rhythm':
        # filter
        counts = pd.Series(np.concatenate(YY.rhythm.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.rhythm = YY.rhythm.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['rhythm_len'] = YY.rhythm.apply(lambda x: len(x))
        # select
        X = XX[YY.rhythm_len > 0]
        Y = YY[YY.rhythm_len > 0]
        mlb.fit(Y.rhythm.values)
        y = mlb.transform(Y.rhythm.values)
    elif ctype == 'all':
        # filter
        counts = pd.Series(np.concatenate(YY.all_scp.values)).value_counts()
        counts = counts[counts > min_samples]
        YY.all_scp = YY.all_scp.apply(lambda x: list(set(x).intersection(set(counts.index.values))))
        YY['all_scp_len'] = YY.all_scp.apply(lambda x: len(x))
        # select
        X = XX[YY.all_scp_len > 0]
        Y = YY[YY.all_scp_len > 0]
        mlb.fit(Y.all_scp.values)
        y = mlb.transform(Y.all_scp.values)
    else:
        pass

    # save LabelBinarizer
    with open(outputfolder+'mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)
    return X, Y, y, mlb

def compute_label_aggregations(df, folder, ctype):

    df['scp_codes_len'] = df.scp_codes.apply(lambda x: len(x))

    aggregation_df = pd.read_csv(folder+'scp_statements.csv', index_col=0)

    if ctype in ['diagnostic', 'subdiagnostic', 'superdiagnostic']:

        def aggregate_all_diagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    tmp.append(key)
            return list(set(tmp))

        def aggregate_subdiagnostic(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_subclass
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        def aggregate_diagnostic(y_dic):    # 提取超类标签
            tmp = []
            for key in y_dic.keys():
                if key in diag_agg_df.index:
                    c = diag_agg_df.loc[key].diagnostic_class
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        diag_agg_df = aggregation_df[aggregation_df.diagnostic == 1.0]
        if ctype == 'diagnostic':
            df['diagnostic'] = df.scp_codes.apply(aggregate_all_diagnostic)
            df['diagnostic_len'] = df.diagnostic.apply(lambda x: len(x))
        elif ctype == 'subdiagnostic':
            df['subdiagnostic'] = df.scp_codes.apply(aggregate_subdiagnostic)
            df['subdiagnostic_len'] = df.subdiagnostic.apply(lambda x: len(x))
        elif ctype == 'superdiagnostic':
            df['superdiagnostic'] = df.scp_codes.apply(aggregate_diagnostic)
            df['superdiagnostic_len'] = df.superdiagnostic.apply(lambda x: len(x))
    elif ctype == 'form':
        form_agg_df = aggregation_df[aggregation_df.form == 1.0]

        def aggregate_form(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in form_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['form'] = df.scp_codes.apply(aggregate_form)
        df['form_len'] = df.form.apply(lambda x: len(x))
    elif ctype == 'rhythm':
        rhythm_agg_df = aggregation_df[aggregation_df.rhythm == 1.0]

        def aggregate_rhythm(y_dic):
            tmp = []
            for key in y_dic.keys():
                if key in rhythm_agg_df.index:
                    c = key
                    if str(c) != 'nan':
                        tmp.append(c)
            return list(set(tmp))

        df['rhythm'] = df.scp_codes.apply(aggregate_rhythm)
        df['rhythm_len'] = df.rhythm.apply(lambda x: len(x))
    elif ctype == 'all':
        df['all_scp'] = df.scp_codes.apply(lambda x: list(set(x.keys())))

    return df

def apply_standardizer(X, ss):
    X_tmp = []
    for x in X:
        x_shape = x.shape
        X_tmp.append(ss.transform(x.flatten()[:,np.newaxis]).reshape(x_shape))
    X_tmp = np.array(X_tmp)
    return X_tmp
def preprocess_signals(X_train, X_validation, X_test, outputfolder):
    # Standardize data such that mean 0 and variance 1
    ss = StandardScaler()
    ss.fit(np.vstack(X_train).flatten()[:, np.newaxis].astype(float))

    # Save Standardizer data
    with open(outputfolder + 'standard_scaler.pkl', 'wb') as ss_file:
        pickle.dump(ss, ss_file)

    return apply_standardizer(X_train, ss), apply_standardizer(X_validation, ss), apply_standardizer(X_test, ss)

def translate_report(report,csv_file_path):
    header = ['index', 'source', 'target']  # Headers of csv file
    prompt_prefix_diagnosis = ("Help me translate the medical report from German into English. Please directly tell me the "
                               "translation result, no other explanatory words. The origin medical report is: ")
    url = "CHAT_WITH_YOUR_GPT"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    error_list = []
    with open(csv_file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)  # Write headers to csv file

        for idx, item in enumerate(report):
            entry = [None] * len(header)  # Initialize an empty list for each entry
            entry[0] = idx  # Remove .hea extension from filename
            entry[1] = item
            try:
                data = {"messages": [
                    {"role": "user", "content": prompt_prefix_diagnosis + item}],
                        "userId": "serveForPaper"}
                json_data = json.dumps(data)
                response = requests.post(url=url, data=json_data, headers=headers)
                json_response = response.json()
                print(idx, json_response["content"])
                entry[2] = json_response["content"].replace("\n\t", "").replace("\n", "")
                writer.writerow(entry)  # Write entry to csv file
                # label_argment[item] = json_response["content"].replace("\n\t", "").replace("\n", "")
            except Exception as e:
                error_list.append(idx)
                entry[2] = "error"
                writer.writerow(entry)
                print(e)
        print(error_list)


def generate_ptb_label_gemini_augment():
    all_label_map = {'NDT': 'non-diagnostic T wave abnormalities',
                     'NST_': 'ST segment changes',
                     'DIG': 'digitalis-effect',
                     'LNGQT': 'long QT interval',
                     'NORM': 'normal ECG',
                     'IMI': 'inferior myocardial infarction',
                     'ASMI': 'anteroseptal myocardial infarction',
                     'LVH': 'left ventricular hypertrophy',
                     'LAFB': 'left anterior fascicular block',
                     'ISC_': 'myocardial ischemic',
                     'IRBBB': 'incomplete right bundle branch block',
                     '1AVB': 'first degree atrioventricular block',
                     'IVCD': 'intraventricular conduction disturbance (block)',
                     'ISCAL': 'anterolateral myocardial ischemic',
                     'CRBBB': 'complete right bundle branch block',
                     'CLBBB': 'complete left bundle branch block',
                     'ILMI': 'inferolateral myocardial infarction',
                     'LAO/LAE': 'left atrial overload/enlargement',
                     'AMI': 'anterior myocardial infarction',
                     'ALMI': 'anterolateral myocardial infarction',
                     'ISCIN': 'inferior myocardial ischemic',
                     'INJAS': 'subendocardial injury in anteroseptal leads',
                     'LMI': 'lateral myocardial infarction',
                     'ISCIL': 'inferolateral myocardial ischemic',
                     'LPFB': 'left posterior fascicular block',
                     'ISCAS': 'anteroseptal myocardial ischemic',
                     'INJAL': 'subendocardial injury in anterolateral leads',
                     'ISCLA': 'lateral myocardial ischemic',
                     'RVH': 'right ventricular hypertrophy',
                     'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
                     'RAO/RAE': 'right atrial overload/enlargement',
                     'EL': 'electrolytic disturbance or drug (former EDIS)',
                     'WPW': 'Wolf-Parkinson-White syndrome',
                     'ILBBB': 'incomplete left bundle branch block',
                     'IPLMI': 'inferoposterolateral myocardial infarction',
                     'ISCAN': 'anterior myocardial ischemic',
                     'IPMI': 'inferoposterior myocardial infarction',
                     'SEHYP': 'septal hypertrophy',
                     'INJIN': 'subendocardial injury in inferior leads',
                     'INJLA': 'subendocardial injury in lateral leads',
                     'PMI': 'posterior myocardial infarction',
                     '3AVB': 'third degree atrioventricular block',
                     'INJIL': 'subendocardial injury in inferolateral leads',
                     '2AVB': 'second degree atrioventricular block',
                     'ABQRS': 'abnormal QRS(QRS changes)',
                     'PVC': 'ventricular premature complex',
                     'STD_': 'ST segment depression',
                     'VCLVH': 'voltage criteria (QRS) for left ventricular hypertrophy',
                     'QWAVE': 'Q waves present',
                     'LOWT': 'low amplitude T wave',
                     'NT_': 'T wave changes',
                     'PAC': 'atrial premature complex',
                     'LPR': 'prolonged PR interval',
                     'INVT': 'inverted T wave',
                     'LVOLT': 'low QRS voltages in the frontal and horizontal leads',
                     'HVOLT': 'high QRS voltage',
                     'TAB_': 'T wave abnormality',
                     'STE_': 'ST segment elevation',
                     'PRC(S)': 'premature complex(es)',
                     'SR': 'sinus rhythm',
                     'AFIB': 'atrial fibrillation',
                     'STACH': 'sinus tachycardia',
                     'SARRH': 'sinus arrhythmia',
                     'SBRAD': 'sinus bradycardia',
                     'PACE': 'normal functioning artificial pacemaker',
                     'SVARR': 'supraventricular arrhythmia',
                     'BIGU': 'bigeminal pattern (unknown origin, SV or Ventricular)',
                     'AFLT': 'atrial flutter',
                     'SVTAC': 'supraventricular tachycardia',
                     'PSVT': 'paroxysmal supraventricular tachycardia',
                     'TRIGU': 'trigeminal pattern (unknown origin, SV or Ventricular)'}
    generated_description_dict = {}
    for item in all_label_map.values():
        response = _generate_gemini_augment_(item)
        print(response)
        generated_description_dict[item] = response
    with open("ptbxl_label_map_description_gemini.json", "w") as f:
        json.dump(generated_description_dict, f)

def _generate_gemini_augment_(item):
    prompt_prefix_zeroshot = "I want you to play the role of a professional Electrocardiologist, and I need you to explain the meaning of "
    prompt_suffix_zeroshot = " in a 12-lead electrocardiogram report. Your answer must be less than 50 words."
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                              "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."
    url = "CAHT_WITH_YOUR_GPT"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    # data = {"messages": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis,
    #         "userId": "serveForPaper"}
    data = {"messages": prompt_prefix_zeroshot + item + prompt_suffix_zeroshot,
            "userId": "serveForPaper"}
    json_data = json.dumps(data)
    response = requests.post(url=url, data=json_data, headers=headers)
    return response.text


def generate_zhipuai_augment():
    """"""
    from zhipuai import ZhipuAI

    prompt_prefix_zeroshot = "I want you to play the role of a professional Electrocardiologist, and I need you to explain the meaning of "
    prompt_suffix_zeroshot = " in a 12-lead electrocardiogram report. Your answer must be less than 50 words."
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                                  "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."

    client = ZhipuAI(api_key="YOUR_ZHIPU_API_KEY")

    all_label_map = {'NDT': 'non-diagnostic T wave abnormalities',
                     'NST_': 'ST segment changes',
                     'DIG': 'digitalis-effect',
                     'LNGQT': 'long QT interval',
                     'NORM': 'normal ECG',
                     'IMI': 'inferior myocardial infarction',
                     'ASMI': 'anteroseptal myocardial infarction',
                     'LVH': 'left ventricular hypertrophy',
                     'LAFB': 'left anterior fascicular block',
                     'ISC_': 'myocardial ischemic',
                     'IRBBB': 'incomplete right bundle branch block',
                     '1AVB': 'first degree atrioventricular block',
                     'IVCD': 'intraventricular conduction disturbance (block)',
                     'ISCAL': 'anterolateral myocardial ischemic',
                     'CRBBB': 'complete right bundle branch block',
                     'CLBBB': 'complete left bundle branch block',
                     'ILMI': 'inferolateral myocardial infarction',
                     'LAO/LAE': 'left atrial overload/enlargement',
                     'AMI': 'anterior myocardial infarction',
                     'ALMI': 'anterolateral myocardial infarction',
                     'ISCIN': 'inferior myocardial ischemic',
                     'INJAS': 'subendocardial injury in anteroseptal leads',
                     'LMI': 'lateral myocardial infarction',
                     'ISCIL': 'inferolateral myocardial ischemic',
                     'LPFB': 'left posterior fascicular block',
                     'ISCAS': 'anteroseptal myocardial ischemic',
                     'INJAL': 'subendocardial injury in anterolateral leads',
                     'ISCLA': 'lateral myocardial ischemic',
                     'RVH': 'right ventricular hypertrophy',
                     'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
                     'RAO/RAE': 'right atrial overload/enlargement',
                     'EL': 'electrolytic disturbance or drug (former EDIS)',
                     'WPW': 'Wolf-Parkinson-White syndrome',
                     'ILBBB': 'incomplete left bundle branch block',
                     'IPLMI': 'inferoposterolateral myocardial infarction',
                     'ISCAN': 'anterior myocardial ischemic',
                     'IPMI': 'inferoposterior myocardial infarction',
                     'SEHYP': 'septal hypertrophy',
                     'INJIN': 'subendocardial injury in inferior leads',
                     'INJLA': 'subendocardial injury in lateral leads',
                     'PMI': 'posterior myocardial infarction',
                     '3AVB': 'third degree atrioventricular block',
                     'INJIL': 'subendocardial injury in inferolateral leads',
                     '2AVB': 'second degree atrioventricular block',
                     'ABQRS': 'abnormal QRS(QRS changes)',
                     'PVC': 'ventricular premature complex',
                     'STD_': 'ST segment depression',
                     'VCLVH': 'voltage criteria (QRS) for left ventricular hypertrophy',
                     'QWAVE': 'Q waves present',
                     'LOWT': 'low amplitude T wave',
                     'NT_': 'T wave changes',
                     'PAC': 'atrial premature complex',
                     'LPR': 'prolonged PR interval',
                     'INVT': 'inverted T wave',
                     'LVOLT': 'low QRS voltages in the frontal and horizontal leads',
                     'HVOLT': 'high QRS voltage',
                     'TAB_': 'T wave abnormality',
                     'STE_': 'ST segment elevation',
                     'PRC(S)': 'premature complex(es)',
                     'SR': 'sinus rhythm',
                     'AFIB': 'atrial fibrillation',
                     'STACH': 'sinus tachycardia',
                     'SARRH': 'sinus arrhythmia',
                     'SBRAD': 'sinus bradycardia',
                     'PACE': 'normal functioning artificial pacemaker',
                     'SVARR': 'supraventricular arrhythmia',
                     'BIGU': 'bigeminal pattern (unknown origin, SV or Ventricular)',
                     'AFLT': 'atrial flutter',
                     'SVTAC': 'supraventricular tachycardia',
                     'PSVT': 'paroxysmal supraventricular tachycardia',
                     'TRIGU': 'trigeminal pattern (unknown origin, SV or Ventricular)'}
    generated_description_dict = {}

    for item in all_label_map.values():
        if item in generated_description_dict.keys():
            continue
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis},
                # {"role": "user", "content": prompt_prefix_zeroshot + item + prompt_suffix_zeroshot},
            ],
        )
        response = response.choices[0].message.content
        print(response)
        generated_description_dict[item] = response

    with open("ptbxl_label_map_report_zhipuai.json", "w") as f:
        json.dump(generated_description_dict, f)



if __name__ == '__main__':
    generate_ptb_label_gemini_augment()

    # generate_zhipuai_augment()
