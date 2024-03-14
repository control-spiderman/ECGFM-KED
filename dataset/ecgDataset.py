# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-09 17:13
import torch
from torch.utils.data import Dataset
import numpy as np
import os
import wfdb
from scipy.signal import resample
import pandas as pd


class MimicivDataset(Dataset):
    def __init__(self, X_data, Y_data, useAugment=False, use_what_label='diagnosis_label', use_what_prompt='base',
                 feature_data=None,useFeature=False, mimic_augment_type="mimiciv_label_map_report"):
        self.X_data = X_data
        self.Y_data = Y_data
        self.mimic_augment_type = mimic_augment_type
        self.background_info, self.label_list = self.get_background_infp()
        self.vis_root = '/home/user/dataSpace/mimic_iv_ecg/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0/'
        # if report is not None:
        report = X_data['report'].values
        if useAugment and not useFeature:
            self.report_data = self.report_augment(report)
        elif useFeature and useAugment:
            # 两者都使用
            self.report_data = self.report_feature_all_augment(report,
                                                           feature_data['features_desc_result'].values)
        elif useFeature:
            self.report_data = self.report_feature_augment(report, feature_data['features_desc_result'].values)
        else:
            self.report_data = [item if not str(item) == 'nan' else "" for item in report]
        # else:
        #     self.report_data = None


    def __len__(self):
        return len(self.X_data)


    def get_background_infp(self):
        mimic_augment_type = self.mimic_augment_type
        with open("/home/user/tyy/project/ked/dataset/mimiciv/"+mimic_augment_type+".json", "r") as f:
            background_info = json.load(f)
        f = open('/home/user/tyy/project/ked/dataset/mimiciv/mlb.pkl', 'rb')
        data = pickle.load(f)
        return background_info, data.classes_

    def report_augment(self, report):
        """"""
        new_report = []
        for idx, item in enumerate(report):
            if str(item) == 'nan':
                new_report.append("")
                continue
            label_list = self.Y_data[idx]
            disease_label_index = np.where(label_list == 1)[0]
            background_list = []
            for sub_idx in disease_label_index:
                sub_label = self.label_list[sub_idx]
                if sub_label in self.background_info.keys():
                    background_list.append(self.background_info[sub_label])
            if background_list:
                background_info = ". ".join(background_list)
                final_report = " This ECG is: " + item + "\nBackground information: " +background_info
            else:
                final_report = " This ECG is: " + item
            new_report.append(final_report)
        return new_report

    def report_feature_augment(self, report, feature):
        """"""
        new_report = []
        for rpt, ftr in zip(report, feature):
            if str(rpt) == 'nan':
                new_report.append("")
                continue
            final_report = ', '.join(ftr) + ". ECG Report: " + rpt
            new_report.append(final_report)
        return new_report

    def report_feature_all_augment(self, report, feature):
        """"""
        new_report = []
        for idx in range(len(report)):
            rpt = report[idx]
            ftr = feature[idx]
            if str(rpt) == 'nan':
                new_report.append("")
                continue
            label_list = self.Y_data[idx]
            disease_label_index = np.where(label_list == 1)[0]
            background_list = []
            for sub_idx in disease_label_index:
                sub_label = self.label_list[sub_idx]
                sub_label = self.all_label_map[sub_label]
                if sub_label in self.background_info.keys():
                    background_list.append(self.background_info[sub_label])
            if background_list:
                background_info = ". ".join(background_list)
                final_report = "Background information: " +background_info+ "ECG feature: " + ', '.join(ftr) +" ECG Report: " + rpt
            else:
                final_report = ', '.join(ftr) + ". ECG Report: " + rpt
            new_report.append(final_report)
        return new_report

    def __getitem__(self, item):
        path = self.X_data["path"].iloc[item]
        ecg_signal = wfdb.rdsamp(os.path.join(self.vis_root, path))  # files/p1205/p12054137/s40989841/40989841
        ecg = ecg_signal[0].T
        ecg = torch.from_numpy(resample(ecg, int(ecg.shape[1] / 5), axis=1))
        if torch.any(torch.isnan(ecg)):
            isnan = torch.isnan(ecg)
            ecg = torch.where(isnan, torch.zeros_like(ecg), ecg)

        disease = self.Y_data[item]
        if self.report_data is not None:
            report = self.report_data[item]
            return {"signal":ecg,  "label":disease, "report": report}
        else:
            return {"signal": ecg, "label": disease}



class NewECGDataset(Dataset):
    def __init__(self, X_data, Y_data, useAugment=True):
        self.X_data = X_data
        self.Y_data = Y_data
        self.useAugment = useAugment
        self.label_name = ["Normal ECG", "Myocardial Infarction", "ST/T change", "Conduction Disturbance",
                           "Hypertrophy"]
        self.label_dict = {"NORM":"Normal ECG", "MI": "Myocardial Infarction", "STTC":"ST/T change", "CD": "Conduction Disturbance", "HYP": "Hypertrophy"}
        self.backgroud_info = {
        "Myocardial Infarction":"To identify myocardial infarction on a 12-lead ECG, focus on leads II, III, and aVF to "
                                "look for ST-segment elevation or depression. Additionally, look for reciprocal changes "
                                "in leads V1-V4. ST-segment elevation in leads V1-V4 may indicate an anterior wall myocardial "
                               "infarction, while ST-segment changes in leads II, III, and aVF may indicate an inferior "
                                "wall myocardial infarction. Q waves may also be present in the affected leads.",
        "ST/T change": "To identify ST/T changes on a 12-lead ECG, the most important leads to focus on are leads II, "
                       "III, aVF, V5, and V6. Look for abnormalities such as ST-segment elevation or depression, T-wave "
                       "inversion or flattening, and QTc prolongation. Pay attention to the morphology and configuration "
                       "of the changes. Other leads may also be helpful, such as lead aVL for detecting lateral wall changes "
                       "and leads V1 and V2 for septal changes. ST-segment depression: ST segment "
                       "below baseline, >1mm below is significant. ST-segment elevation: ST segment above baseline, >1mm above is significant.",
        "Conduction Disturbance":"In identifying conduction disturbances from a 12-lead ECG, you need to focus on the PR "
                                 "interval and the QRS duration.  A prolonged PR interval indicates first-degree AV block "
                                 "while a short PR interval suggests a possible Wolff-Parkinson-White (WPW) syndrome. "
                                 "A widened QRS can indicate bundle branch block, while a narrow QRS suggests normal conduction. "
                                 "First degree AV block: Prolonged PR interval (>200ms). Left bundle branch block: Wide QRS (>120ms) with slurred R wave in V6. "
                                 "Right bundle branch block: Wide QRS (>120ms) with rabbit ears in V1.",
        "Hypertrophy": "To identify hypertrophy from a 12-lead ECG, you should focus on the QRS complex.  Specifically, "
                      "look for an increase in the amplitude of the QRS complex, which can suggest ventricular hypertrophy. "
                      "You should also examine leads V1 and V2, as a prominent R wave in these leads may indicate right "
                      "ventricular hypertrophy, while a prominent S wave in leads V5 and V6 may suggest left ventricular "
                      "hypertrophy.  Be sure to compare the amplitudes of the QRS complexes across all leads to make a "
                      "definitive diagnosis."}
        self.prompt = ["I want you to play the role of a professional electrocardiologist, and I need you to teach me how "
                       "to identify abnormal electrocardiograms. How to identify Hypertrophy from 12-lead ECG, such as what "
                       "leads or what features to focus on ,etc. Your answer must be less than 100 words."]
        feature_data, self.old_report, self.old_label_data = self.importData()
        self.feature_data = torch.FloatTensor(feature_data.astype('float64'))
        self.label_data = self._label_map(self.old_label_data)
        if self.useAugment:
            self.report_data = self.report_augment(self.old_report)
        else:
            self.report_data = self.old_report


    def importData(self):
        return self.X_data, self.Y_data[:, :1], self.Y_data[:, 1:2]


    def __len__(self):
        return len(self.feature_data)

    def disease_idx2name(self, idx):
        """"""
        return self.label_name[idx]

    def __label_map_one(self, old_label):
        label_dict = {"NORM":"Normal ECG", "MI": "Myocardial Infarction", "STTC":"ST/T change", "CD": "Conduction Disturbance", "HYP": "Hypertrophy"}
        label_list = [0, 0, 0, 0, 0]
        for item in old_label[0]:
            new_label = label_dict[item]
            label_list[self.label_name.index(new_label)] = 1
        return label_list

    def _label_map(self, label_data):
        total_label = []
        for item in label_data:
            total_label.append(self.__label_map_one(item))
        return np.array(total_label).astype('int32')

    def report_augment(self, old_report):
        """"""
        new_report = []
        for idx, item in enumerate(old_report):
            label_list = self.old_label_data[idx]
            background_list = []
            for sub_idx in label_list[0]:
                sub_label = self.label_dict[sub_idx]
                if sub_label in self.backgroud_info.keys():
                    background_list.append(self.backgroud_info[sub_label])
            if background_list:
                background_info = ". ".join(background_list)
                final_report = "Background information: " +background_info+ " This ECG is: " + item[0]
            else:
                final_report = " This ECG is: " + item[0]
            new_report.append(final_report)
        return new_report


    def __getitem__(self, item):
        signal = self.feature_data[item]
        if self.useAugment:
            report = self.report_data[item]
        else:
            report = self.report_data[item][0]
        disease = self.label_data[item]
        disease_name = ""
        return {"signal":signal, "report":report, "label":disease, "label_name":disease_name}

import json, pickle
class TotalLabelDataset(Dataset):
    def __init__(self, X_data, Y_data, report=None, useAugment=False, use_what_label='diagnosis_label', use_what_prompt='base',
                 feature_data=None,useFeature=False):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.background_info, self.all_label_map = self.get_background_infp(use_what_label, use_what_prompt)
        self.label_list = self.get_label_list(use_what_label)
        if report is not None:
            if useAugment and not useFeature:
                self.report_data = self.report_augment(report["target"].values)
            elif useFeature and useAugment:
                # 两者都使用
                self.report_data = self.report_feature_all_augment(report["target"].values,
                                                               feature_data['features_desc_result'].values)
            elif useFeature:
                self.report_data = self.report_feature_augment(report["target"].values, feature_data['features_desc_result'].values)
            else:
                self.report_data = [item if not str(item) == 'nan' else "" for item in report["target"].values]
        else:
            self.report_data = None


    def __len__(self):
        return len(self.X_data)

    def get_label_list(self, use_what_label):
        # /home/user/tyy/project/ked or /home/tyy/project/ecgfm_ked
        if use_what_label == 'diagnosis_label':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/mlb.pkl', 'rb')
        elif use_what_label == 'all':
            f = open('/home/tyy/project/ecgfm_ked/dataset/ptb-xl/output/exp0/data/mlb.pkl', 'rb')
        elif use_what_label == 'subdiagnosis_label':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        return data.classes_

    def get_background_infp(self, use_what_label, use_what_prompt="base"):
        if use_what_label in ["diagnosis_label", "all"]:
            if use_what_prompt == 'base':
                background_path = "/home/tyy/project/ecgfm_ked/dataset/all_label_augment.json"
            elif use_what_prompt == 'concise':
                background_path = "/home/user/tyy/project/ked/dataset/prompt_label/label_map_concise.json"
            elif use_what_prompt == 'plain_diagnosis':
                background_path = "/home/user/tyy/project/ked/dataset/prompt_label/label_map_plain_diagnosis.json"
            elif use_what_prompt == 'intern':
                background_path = "/home/user/tyy/project/ked/dataset/prompt_label/label_map_intern.json"
            with open(background_path, 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/project/ecgfm_ked/dataset/all_label_map.json", 'r') as f:
                all_label_map = json.load(f)

            return background_info, all_label_map
        elif use_what_label == "subdiagnosis_label":
            with open("/home/user/tyy/project/ked/dataset/label_augment_23.json", 'r') as f:
                background_info = json.load(f)
            all_label_map = {'NORM':"normal ECG",
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
            return background_info, all_label_map

    def report_augment(self, report):
        """"""
        new_report = []
        for idx, item in enumerate(report):
            if str(item) == 'nan':
                new_report.append("")
                continue
            label_list = self.Y_data[idx]
            disease_label_index = np.where(label_list == 1)[0]
            background_list = []
            for sub_idx in disease_label_index:
                sub_label = self.label_list[sub_idx]
                sub_label = self.all_label_map[sub_label]
                if sub_label in self.background_info.keys():
                    background_list.append(self.background_info[sub_label])
            if background_list:
                background_info = ". ".join(background_list)
                final_report = "Background information: " +background_info+ " This ECG is: " + item
            else:
                final_report = " This ECG is: " + item
            new_report.append(final_report)
        return new_report

    def report_feature_augment(self, report, feature):
        """"""
        new_report = []
        for rpt, ftr in zip(report, feature):
            if str(rpt) == 'nan':
                new_report.append("")
                continue
            final_report = ', '.join(ftr) + ". ECG Report: " + rpt
            new_report.append(final_report)
        return new_report

    def report_feature_all_augment(self, report, feature):
        """"""
        new_report = []
        for idx in range(len(report)):
            rpt = report[idx]
            ftr = feature[idx]
            if str(rpt) == 'nan':
                new_report.append("")
                continue
            label_list = self.Y_data[idx]
            disease_label_index = np.where(label_list == 1)[0]
            background_list = []
            for sub_idx in disease_label_index:
                sub_label = self.label_list[sub_idx]
                sub_label = self.all_label_map[sub_label]
                if sub_label in self.background_info.keys():
                    background_list.append(self.background_info[sub_label])
            if background_list:
                background_info = ". ".join(background_list)
                final_report = "Background information: " +background_info+ "ECG feature: " + ', '.join(ftr) +" ECG Report: " + rpt
            else:
                final_report = ', '.join(ftr) + ". ECG Report: " + rpt
            new_report.append(final_report)
        return new_report

    def __getitem__(self, item):
        signal = self.X_data[item]
        disease = self.Y_data[item]
        if self.report_data is not None:
            report = self.report_data[item]
            return {"signal":signal,  "label":disease, "report": report}
        else:
            return {"signal": signal, "label": disease}

class ICBEBDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.label_name = ["Normal ECG", "First-degree atrioventricular block","Left bundle branch block",
                           "Right bundle branch block", "ST-segment depression", "ST-segment elevated"]

    def disease_idx2name(self, idx):
        return self.label_name[idx]

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, item):
        signal = self.X_data[item]
        label = self.Y_data[item]
        return {"signal":signal, "label":label}

class FinetuneDataset(Dataset):
    def __init__(self, X_data, Y_data, dataset_type, report_data=None, label_type=None, isFinetune=False,
                 zeroshot_report_type=""):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.origin_report_data = report_data
        self.dataset_type = dataset_type
        self.label_type = label_type
        self.finetune = isFinetune
        self.zeroshot_report_type = zeroshot_report_type
        if self.dataset_type == 'georgia':
            self.label_name, self.report_dict = self.load_georgia_label_and_report()
        elif self.dataset_type == 'shaoxing':
            self.label_name, self.report_dict = self.load_shaoxing_label_and_report()
        elif self.dataset_type == 'cpsc':
            self.label_name, self.report_dict = self.load_cpsc_label_and_report()
        elif self.dataset_type == 'clinical':
            self.label_name, self.report_dict = self.load_clinical_label_and_report()
        elif self.dataset_type == 'code_test':
            self.label_name, self.report_dict = self.load_code_label_and_report()
        elif self.dataset_type == 'ptb-xl':
            self.label_name, self.report_dict = self.load_ptbxl_label_and_report()
        if isFinetune:
            self.report_data = self.generate_label_report()
        self._gpt4_unseen_label_desc_without_ptbxl = {
            'Supraventricular Tachycardia': "Supraventricular Tachycardia (SVT) in a 12-lead ECG report refers to a rapid heart rate originating above the ventricles, typically seen as narrow complex tachycardia. It's characterized by abrupt onset and termination, often with P waves hidden in preceding T waves.",
            'Atrial Tachycardia': "Atrial Tachycardia in a 12-lead ECG report indicates a fast heart rate originating from the atria, above the ventricles. It's characterized by a heart rate of over 100 beats per minute and distinct P waves preceding each QRS complex.",
            'ST segment depression': "ST segment depression in a 12-lead ECG report indicates possible myocardial ischemia, a condition where the heart muscle isn't getting enough oxygen. It's often associated with conditions like angina or a heart attack.",
            'inverted T-waves': "Inverted T-waves in a 12-lead ECG report often indicate myocardial ischemia, a condition where the heart muscle isn't getting enough oxygen. However, they can also be seen in other conditions like electrolyte imbalance, or be a normal variant in some individuals.",
            'Q wave abnormal': 'An abnormal Q wave in a 12-lead ECG report often indicates past myocardial infarction (heart attack). It signifies that a portion of the heart muscle has experienced permanent damage due to insufficient blood supply.'}

    def load_ptbxl_label_and_report(self):
        with open("/home/user/tyy/project/ked/dataset/all_label_map_2_8.json", 'r') as f:
            all_label_map = json.load(f)
        #最新修改过的描述使用下面这个：
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
        if self.label_type == 'form':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp2/data/mlb.pkl', 'rb')
        elif self.label_type == 'rhythm':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp3/data/mlb.pkl', 'rb')
        elif self.label_type == 'all':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/mlb.pkl', 'rb')
        elif self.label_type == 'diagnosis_label':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/mlb.pkl', 'rb')
        elif self.label_type == 'subdiagnosis_label':
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/mlb.pkl', 'rb')
            with open("/home/user/tyy/project/ked/dataset/.json", 'r') as f:
                all_label_map = json.load(f)
        else:
            f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1.1/data/mlb.pkl', 'rb')
            all_label_map = {
                "CD": "In identifying conduction disturbances from a 12-lead ECG, you need to focus on the PR "
                      "interval and the QRS duration.  A prolonged PR interval indicates first-degree AV block "
                      "while a short PR interval suggests a possible Wolff-Parkinson-White (WPW) syndrome. "
                      "A widened QRS can indicate bundle branch block, while a narrow QRS suggests normal conduction. "
                      "First degree AV block: Prolonged PR interval (>200ms). Left bundle branch block: Wide QRS (>120ms) with slurred R wave in V6. "
                      "Right bundle branch block: Wide QRS (>120ms) with rabbit ears in V1.",
                "HYP": "Hypertrophy(HYP)",
                "MI": "Myocardial Infarction(MI)",
                "NORM": "normal ECG(NORM)",
                "STTC": "ST/T changes"
            }
        data = pickle.load(f)
        if self.zeroshot_report_type == "gemini_desc":
            label_report_file = '/home/user/tyy/project/ked/dataset/ptb-xl/ptbxl_label_map_description_gemini.json' # 这个是gemini描述版本的
        elif self.zeroshot_report_type == "gemini_report":
            label_report_file = '/home/user/tyy/project/ked/dataset/ptb-xl/ptbxl_label_map_report_gemini.json' # 这个是gemini报告版本的
        elif self.zeroshot_report_type == "zhipuai_desc":
            label_report_file = '/home/user/tyy/project/ked/dataset/ptb-xl/ptbxl_label_map_description_zhipuai.json' # 这个是glm4描述版本的
        elif self.zeroshot_report_type == "zhipuai_report":
            label_report_file = '/home/user/tyy/project/ked/dataset/ptb-xl/ptbxl_label_map_report_zhipuai.json' # 这个是glm4报告版本的
        elif self.zeroshot_report_type == "gpt4_desc":
            label_report_file = '/home/user/tyy/project/ked/dataset/ptb-xl/ptbxl_label_map_description_gpt.json'  # 这个是gpt-4报告版本的
        else:
            label_report_file = '/home/user/tyy/project/ked/dataset/all_label_augment_2_9.json'  # 这个是gpt-4报告版本的

        with open(label_report_file, 'r') as f:

            label_report = json.load(f)

        label_seen_type = {
  "first degree atrioventricular block": 1,
  "second degree atrioventricular block": 0,
  "third degree atrioventricular block": 0,
  "abnormal QRS(QRS changes)": 0,
  "atrial fibrillation": 1,
  "atrial flutter": 1,
  "anterolateral myocardial infarction": 1,
  "anterior myocardial infarction": 1,
  "ST-T changes compatible with ventricular aneurysm": 0,
  "anteroseptal myocardial infarction": 1,
  "bigeminal pattern (unknown origin, SV or Ventricular)": 0,
  "complete left bundle branch block": 2,
  "complete right bundle branch block": 2,
  "digitalis-effect": 0,
  "electrolytic disturbance or drug (former EDIS)": 0,
  "high QRS voltage": 0,
  "incomplete left bundle branch block": 1,
  "inferolateral myocardial infarction": 0,
  "inferior myocardial infarction": 1,
  "subendocardial injury in anterolateral leads": 0,
  "subendocardial injury in anteroseptal leads": 0,
  "subendocardial injury in inferolateral leads": 0,
  "subendocardial injury in inferior leads": 0,
  "subendocardial injury in lateral leads": 0,
  "inverted T wave": 0,
  "inferoposterolateral myocardial infarction": 0,
  "inferoposterior myocardial infarction": 0,
  "incomplete right bundle branch block": 1,
  "anterolateral myocardial ischemic": 2,
  "anterior myocardial ischemic": 2,
  "anteroseptal myocardial ischemic": 2,
  "inferolateral myocardial ischemic": 2,
  "inferior myocardial ischemic": 2,
  "lateral myocardial ischemic": 2,
  "myocardial ischemic": 1,
  "intraventricular conduction disturbance (block)": 1,
  "left anterior fascicular block": 1,
  "left atrial overload/enlargement": 2,
  "lateral myocardial infarction": 1,
  "long QT interval": 2,
  "low amplitude T wave": 0,
  "left posterior fascicular block": 0,
  "prolonged PR interval": 1,
  "left ventricular hypertrophy": 1,
  "low QRS voltages in the frontal and horizontal leads": 2,
  "non-diagnostic T wave abnormalities": 2,
  "normal ECG": 1,
  "ST segment changes": 0,
  "T wave changes": 2,
  "atrial premature complex": 1,
  "normal functioning artificial pacemaker": 0,
  "posterior myocardial infarction": 0,
  "premature complex(es)": 0,
  "paroxysmal supraventricular tachycardia": 0,
  "ventricular premature complex": 1,
  "Q waves present": 0,
  "right atrial overload/enlargement": 0,
  "right ventricular hypertrophy": 1,
  "sinus arrhythmia": 1,
  "sinus bradycardia": 1,
  "septal hypertrophy": 2,
  "sinus rhythm": 1,
  "sinus tachycardia": 1,
  "ST segment depression": 0,
  "ST segment elevation": 1,
  "supraventricular arrhythmia": 0,
  "supraventricular tachycardia": 0,
  "T wave abnormality": 1,
  "trigeminal pattern (unknown origin, SV or Ventricular)": 0,
  "voltage criteria (QRS) for left ventricular hypertrophy": 2,
  "Wolf-Parkinson-White syndrome": 0
}
        form = ["abnormal QRS(QRS changes)"
                ,"digitalis-effect"
                ,"high QRS voltage"
                ,"inverted T wave"
                ,"long QT interval"
                ,"low amplitude T wave"
                ,"prolonged PR interval"
                ,"low QRS voltages in the frontal and horizontal leads"
                ,"non-diagnostic T wave abnormalities"
                ,"ST segment changes"
                ,"T wave changes"
                ,"atrial premature complex"
                ,"premature complex(es)"
                ,"ventricular premature complex"
                ,"Q waves present"
                ,"ST segment depression"
                ,"ST segment elevation"
                ,"T wave abnormality"
                ,"voltage criteria (QRS) for left ventricular hypertrophy"]
        rhythm = ["atrial fibrillation",
            "atrial flutter",
            "bigeminal pattern (unknown origin, SV or Ventricular)",
            "normal functioning artificial pacemaker",
            "paroxysmal supraventricular tachycardia",
            "sinus arrhythmia",
            "sinus bradycardia",
            "sinus rhythm",
            "sinus tachycardia",
            "supraventricular arrhythmia",
            "supraventricular tachycardia",
            "trigeminal pattern (unknown origin, SV or Ventricular)",
        ]
        final_label = [label_report[all_label_map[item]] if label_seen_type[all_label_map[item]] == 0
                                                            # and
                                                            # (all_label_map[item] in form
                                                            #  or all_label_map[item] in rhythm
                                                            #  )
                       else all_label_map[item] for item in data.classes_ ]
        # final_label = [all_label_map[item] for item in data.classes_]

        return final_label, label_report

    def load_georgia_label_and_report(self):
        with open("/home/user/tyy/project/ked/dataset/georgia/label_map.json", 'r') as f:
            all_label_map = json.load(f)
        f = open('/home/user/tyy/project/ked/dataset/georgia/mlb.pkl', 'rb')
        data = pickle.load(f)
        with open("/home/user/tyy/project/ked/dataset/georgia/georgia_label_map_report.json", 'r') as f:
            label_report = json.load(f)
        # return [all_label_map[item] for item in data.classes_], label_report
        label_name = ['Atrial fibrillation',
                      'Atrial flutter',
                      'first degree AV block',
                      'Incomplete right bundle branch block',
                      'Left axis deviation',
                      'Left anterior fascicular block',
                      'Left bundle branch block',
                      'Low QRS voltages:Look for QRS amplitudes less than 5 mm in limb leads and less than 10 mm in precordial leads.',
                      'long QT-interval:Look for a prolonged QT interval (>440 ms in males, >460 ms in females) and assess for T-wave abnormalities, such as T-wave notching or low-amplitude T-waves.',
                      'non-specific intraventricular conduction disturbance (block)',
                      'Sinus rhythm',
                      "Atrial premature complex(APC):Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
                      'Q wave abnormal(QAb):Focus on leads II, III, aVF, V5, and V6. Look for Q waves deeper than 1/3 the R wave height, lasting longer than 0.04 seconds, and present in at least two contiguous leads. Seek medical advice for accurate interpretation.',
                      'Right bundle branch block',
                      "Sinus arrhythmia(SA):Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
                      'Sinus bradycardia',
                      'Sinus tachycardia',
                      'T wave abnormal:Look for inverted or peaked T-waves, asymmetry, or changes in amplitude. Compare with other leads for confirmation. Seek medical advice for accurate interpretation and diagnosis.',
                      'inverted T-waves',
                      'ventricular premature complex']
        label_desc_map = {'Atrial fibrillation': 'Look for irregularly irregular rhythm, absence of P waves, and variable ventricular response in the ECG. Leads II, III, and aVF often provide the clearest view of these features.',
                          'Atrial flutter': 'Look for regular atrial activity at around 300 bpm in leads II, III, and aVF. The classic "sawtooth" pattern is a key feature. Ventricular rate is often around 150 bpm. Absence of P waves and presence of F waves are also indicative.',
                          'first degree atrioventricular block': 'Look for a prolonged PR interval (>200ms) in all leads of the ECG. This is the key feature of a first degree AV block. The P wave and QRS complex remain consistent, but the delay occurs between them.',
                          'Incomplete right bundle branch block': "Look for RSR' pattern in V1-V3 leads in the ECG. The QRS complex duration should be less than 120 ms. The R' wave in V1 or V2 should be smaller than the initial R wave.",
                          'Left axis deviation': "Check leads I and aVF on the ECG. If the QRS complex is negative in lead aVF and positive in lead I, it indicates Left Axis Deviation. This is due to the heart's electrical axis shifting more to the left than normal.",
                          'Left anterior fascicular block': 'Look for left axis deviation (between -45 and -90 degrees) in the frontal plane, normal or slightly widened QRS complex, and small Q waves in leads I and aVL on the 12-lead ECG to diagnose Left Anterior Fascicular Block.',
                          'Left bundle branch block': "Look for broad QRS complex (>120 ms) in leads I, V5, and V6 on ECG. The QRS complex will have a 'M' shape in V5-V6 and a 'W' shape in V1-V2. This indicates a delay or blockage in electrical conduction in the left bundle branch.",
                          'Low QRS voltages': 'Low QRS voltages are diagnosed when the amplitude of the QRS complex is less than 5mm in all limb leads or less than 10mm in all precordial leads on a 12-lead ECG.',
                          'long QT-interval': "Look at leads II, V5, or V6 on the ECG. Measure from the start of the Q wave to the end of the T wave (QT interval). If it's more than half the distance between two R waves (R-R interval), it's a long QT.",
                          'non-specific intraventricular conduction disturbance (block)': "Look for widened QRS complex (>120 ms) in the ECG. It's not localized to a specific bundle branch, hence called non-specific. It may be due to myocardial diseases, electrolyte imbalances, or drug toxicity. Always correlate with clinical symptoms and history.",
                          'Sinus rhythm': 'Look for regular P waves preceding each QRS complex in lead II. Check for a consistent PR interval (0.12-0.20 sec) and a normal QRS duration (<0.12 sec). P wave axis should be between 0°-+75°. These indicate normal sinus rhythm.',
                          'Atrial premature complex': "Look for irregular P waves that occur earlier than expected in any lead. These P waves may have a different shape, indicating they originate outside the sinus node. The following QRS complex is usually normal unless there's an underlying conduction defect.",
                          'Q wave abnormal': 'Look for Q waves in leads II, III, aVF (inferior wall) and V1-V6 (anterior and lateral walls). Abnormal Q waves are >40ms wide, >2mm deep, or >25% of the following R wave. They indicate past myocardial infarction.',
                          'Right bundle branch block': "Look for a widened QRS complex (>0.12 sec) in leads V1-V3 on the ECG. The QRS complex will have a 'rabbit ear' appearance (R, R', S pattern) in these leads. This indicates a delay in electrical conduction through the right bundle branch, known as Right Bundle Branch Block.",
                          'Sinus arrhythmia': "Look for regular variation in R-R intervals on ECG. In sinus arrhythmia, P waves are normal but R-R intervals vary. It's often associated with breathing cycles. No specific lead is superior, but lead II often provides a good view.",
                          'Sinus bradycardia': 'Look for a regular rhythm with a heart rate less than 60 bpm on the ECG. P waves should be present and normal, indicating the impulse is originating from the sinus node. Check all 12 leads, but lead II gives a good view of P waves.',
                          'Sinus tachycardia': 'Sinus tachycardia on a 12-lead ECG is diagnosed by a heart rate over 100 bpm, regular rhythm, and P waves preceding each QRS complex. Focus on leads II, III, and aVF for clear P wave visibility.',
                          'T wave abnormal': 'Look for T wave inversion in leads V1-V4, which may indicate ischemia. T wave flattening or inversion in II, III, aVF, or V4-V6 could suggest coronary artery disease. Tall, peaked T waves may indicate hyperkalemia. Always compare with previous ECGs if available.',
                          # 'inverted T-waves': "Inverted T-waves can be diagnosed by focusing on leads V1-V4. Look for a negative deflection after the QRS complex. However, T-wave inversion can be normal in leads III, aVR, and V1. Always compare with previous ECGs and consider patient's clinical context.",
                          "Inverted T-waves can be diagnosed by focusing on leads V1-V4. Look for a negative deflection after the QRS complex. However, T-wave inversion can be normal in leads III, aVR, and V1. Always compare with previous ECGs and consider patient's clinical context.": "Inverted T-waves can be diagnosed by focusing on leads V1-V4. Look for a negative deflection after the QRS complex. However, T-wave inversion can be normal in leads III, aVR, and V1. Always compare with previous ECGs and consider patient's clinical context.",
                          'ventricular premature complex': 'Look for premature beats with wide QRS complex (>120 ms) in the ECG. They often have a bizarre morphology, with a right or left bundle branch block pattern. The T wave usually has an opposite direction to the main vector of the QRS complex.'}
        # new_class = ['inverted T-waves', 'Q wave abnormal']
        return [key for key, value in label_desc_map.items()], label_report
        # return [key + '. ' + value for key, value in label_desc_map.items()], label_report

    def load_shaoxing_label_and_report(self):
        all_label_map = {"SB": "Sinus Bradycardia", "SR": "Sinus Rhythm", "AFIB": "Atrial Fibrillation",
                         "ST": "Sinus Tachycardia", "AF": "Atrial Flutter", "SA": "Sinus Arrhythmia",
                         "SVT": "Supraventricular Tachycardia", "AT": "Atrial Tachycardia"}
        # all_label_map = {"SB": "Sinus Bradycardia:Look for a regular rhythm with a heart rate below 60 bpm. The P waves should be upright and consistent, with a normal PR interval.",
        #                  "SR": "Sinus Rhythm",
        #                  "AFIB": "Atrial Fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
        #                  "ST": "Sinus Tachycardia(ST):Look for a regular rhythm, upright P waves preceding each QRS complex, and a heart rate above 100 beats per minute.",
        #                  "AF": "Atrial Flutter(AF):Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
        #                  "SA": "Sinus Arrhythmia(SA):Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
        #                  "SVT": "Supraventricular Tachycardia(SVT):Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
        #                  "AT": "Atrial Tachycardia(AT):Look for a regular narrow QRS complex rhythm with a heart rate >100 bpm. P-waves may be hidden in the preceding T-wave or may appear abnormal, often with a different morphology compared to sinus rhythm."}
        f = open('/home/user/tyy/project/ked/dataset/shaoxing/mlb.pkl', 'rb')
        data = pickle.load(f)
        # /home/tyy/project/ecgfm_ked
        with open("/home/user/tyy/project/ked/dataset/shaoxing/shaoxing_label_map_report.json", 'r') as f:
            label_report = json.load(f)
        label_desc_map = {'Sinus Bradycardia': 'Look for a regular rhythm with a heart rate less than 60 bpm on the ECG. P waves should be present and normal, indicating the impulse is originating from the sinus node. Check all 12 leads, but lead II gives a good view of P waves.',
                          'Sinus Rhythm': 'Look for regular P waves preceding each QRS complex in all leads. The P wave should be upright in leads I, II, aVF and V2-V6. The rate should be between 60-100 beats per minute. The PR interval should be consistent and between 120-200 ms.',
                          'Atrial Fibrillation': 'Look for irregularly irregular rhythm, absence of P waves, and variable ventricular response in the ECG. Primarily focus on leads II, III, and aVF for atrial activity.',
                          'Sinus Tachycardia': 'Sinus Tachycardia on a 12-lead ECG is diagnosed by a heart rate over 100 bpm, regular rhythm, and P waves preceding each QRS complex. Focus on leads II, III, and aVF for clear P wave visibility.',
                          'Atrial Flutter': 'Look for regular atrial activity at around 300 bpm in leads II, III, and aVF. The classic "sawtooth" pattern is a key feature. Ventricular rate is often around 150 bpm. Absence of P waves and presence of F waves are also indicative.', 'Sinus Arrhythmia': "Look for a regular variation in the R-R interval on the ECG. If the heart rate increases during inhalation and decreases during exhalation, it's Sinus Arrhythmia. It's usually seen in all leads.", 'Supraventricular Tachycardia': "Look for a rapid heart rate (>100 bpm) on the ECG. In SVT, P waves may be hard to see. If visible, they may appear abnormal or be located just after the QRS complex. The QRS complex is usually narrow unless there's a conduction abnormality.", 'Atrial Tachycardia': 'Look for a heart rate over 100 bpm, P waves before each QRS complex, and abnormal P wave morphology in leads II, III, and aVF. P wave axis may also be abnormal. The PR interval may be short if the atrial rate is very fast.'}
        # return [all_label_map[item] + '.  background info:'+ label_desc_map[all_label_map[item]] for item in data.classes_], label_report
        return [all_label_map[item] for item in data.classes_], label_report

    def load_cpsc_label_and_report(self):
        label_list = ['normal ECG', 'Atrial fibrillation', 'first degree AV block', 'left bundle branch block','Right bundle branch block',
                                "atrial premature complex","ventricular premature complex",'non-specific ST depression','non-specific ST elevation']
        # label_list = ['normal ECG',
        #               'Atrial fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.',
        #               'first degree AV block(IAVB):focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.',
        #               'left bundle branch block(LBBB):Look for a wide QRS complex (>120 ms), broad R waves, and deep S waves in these leads. Additionally, observe the absence of Q waves in leads I, aVL, V5, and V6.',
        #               "Right bundle branch block(RBBB):Look for a wide QRS complex (>0.12 seconds) with an rSR' pattern in these leads. The presence of a slurred S wave in leads I, aVL, and V6 can also indicate RBBB.",
        #               "Atrial premature complex(APC):Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
        #               "Ventricular premature complex(VPC):Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
        #               'non-specific ST depression(STD_):Look for downsloping or horizontal ST segment depression ≥1mm below the isoelectric line, often accompanied by T wave inversion.',
        #               'non-specific ST elevation(STE_):Look for an elevation of the ST segment above the baseline of at least 1mm in two consecutive leads.']
        # label_list = ['normal ECG',
        #               'Atrial fibrillation(AFIB)',
        #               'first degree AV block(IAVB)',
        #               'left bundle branch block(LBBB)',
        #               "Right bundle branch block(RBBB)",
        #               "Atrial premature complex(APC)",
        #               "Ventricular premature complex(VPC)",
        #               'ST segment depression',
        #               'ST segment elevation']
        label_desc_map = {'normal ECG': 'Focus on P wave, QRS complex, and T wave. Check rate and rhythm. Look for abnormalities in PR interval, QRS duration, and QT interval. Examine ST segment and T wave for ischemia or infarction. Review each of the 12 leads for a comprehensive view.',
                          'Atrial fibrillation': 'Look for irregularly irregular rhythm, absence of P waves, and variable ventricular response in the ECG. Leads II, III, and aVF often show these features best. Also, note the presence of fibrillatory waves instead of normal P waves.',
                          'first degree atrioventricular block': 'Look for a prolonged PR interval (>200ms) in all leads of the ECG. This is the key feature of a first degree AV block.',
                          'left bundle branch block': "Look for broad QRS complex (>120 ms) in leads I, V5, and V6 on ECG. The QRS complex will have a 'M' shape in V5-V6 and a 'W' shape in V1-V2. This indicates a delay or blockage in electrical signals on the left side of the heart.",
                          'Right bundle branch block': "Look for a widened QRS complex (>0.12 sec) in leads V1-V3 on the ECG. The QRS complex will have a 'rabbit ear' appearance (R, R') in these leads. This indicates a delay in electrical conduction through the right bundle branch, hence Right Bundle Branch Block.",
                          'Atrial premature complex': 'Look for irregular P waves that occur earlier than expected in any lead. These P waves may have a different shape compared to normal ones, indicating they originate from a different part of the atria. This is followed by a normal QRS complex.',
                          'Ventricular premature complex': 'Look for premature beats with wide QRS complex (>120 ms) in the ECG. They often have a bizarre morphology, with a right or left bundle branch block pattern. The T wave usually has an opposite direction to the main vector of the QRS complex.',
                          'ST segment depression': "Look for horizontal or downsloping ST segment depression ≥ 0.5mm in two contiguous leads in 12-lead ECG. It's often seen in leads V4-V6, II, III, aVF. It may indicate myocardial ischemia, digoxin effect, or ventricular hypertrophy.",
                          # "Look for horizontal or downsloping ST segment depression ≥ 0.5mm in two contiguous leads in 12-lead ECG. It's often seen in leads V4-V6, II, III, aVF. It may indicate myocardial ischemia, digoxin effect, or ventricular hypertrophy.": "Look for horizontal or downsloping ST segment depression ≥ 0.5mm in two contiguous leads in 12-lead ECG. It's often seen in leads V4-V6, II, III, aVF. It may indicate myocardial ischemia, digoxin effect, or ventricular hypertrophy.",
                          'ST segment elevation': 'Look for a J-point elevation of at least 1mm in two contiguous leads. The ST segment should be concave or horizontal. Check leads V1-V6, II, III, aVF for anterior and inferior wall MI. Leads I, aVL, V5, V6 for lateral wall MI.'}
        with open("/home/user/tyy/project/ked/dataset/cpsc/cpsc_label_map_report.json", 'r') as f:
            label_report = json.load(f)
        # return [key for key, value in label_desc_map.items()], label_report
        return [key for key, value in label_desc_map.items()], label_report

    def load_code_label_and_report(self):
        label_list = ['1dAVb', 'RBBB', 'LBBB', 'SB', 'AF', 'ST']
        label_dict = {
            '1dAVb': "first degree atrioventricular block",
            "RBBB": "right bundle branch block",
            'LBBB': "left bundle branch block",
            "AF": "Atrial Fibrillation",
            "SB": "Sinus Bradycardia",
            "ST": "Sinus Tachycardia"}
        # f = open('/home/user/tyy/project/ked/dataset/clinical_dataset/mlb12.pkl', 'rb')
        # data = pickle.load(f)
        return [label_dict[item] for item in label_list], None

    def load_clinical_label_and_report(self):
        label_list = {
            'I°房室传导阻滞': "first degree atrioventricular block",
            'T波倒置': "inverted T wave",
            "不齐": "Sinus arrhythmia",
            "完全性右束支传导阻滞": "complete right bundle branch block",
            '完全性左束支传导阻滞': "complete left bundle branch block",
            "室上性心动过速": "Supraventricular Tachycardia",
            # "室性早搏": "ventricular premature complex:Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
            "室性早搏": "ventricular premature complex",
            "左前分支传导阻滞": "Left anterior fascicular block",
            "心房扑动": "Atrial Flutter",
            "心房颤动": "Atrial Fibrillation",
            # "房性早搏": "atrial premature complex:Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
            "房性早搏": "atrial premature complex",
            "正常心电图": "normal ECG",
            "窦性心动过缓": "Sinus Bradycardia",
            "窦性心动过速": "Sinus Tachycardia"}
        label_desc_map = {
            'first degree AV block': 'Look for a prolonged PR interval (>200ms) in all leads of the ECG. This is the key feature of a first degree AV block. The P wave and QRS complex remain consistent, but the time between them increases.',
            'inverted T-waves': 'Inverted T-waves can be diagnosed by focusing on leads V1-V6, II, III, and aVF in a 12-lead ECG. Look for a negative deflection after the QRS complex, which indicates an inverted T-wave, suggesting myocardial ischemia or ventricular strain.',
            'Sinus arrhythmia': "Look for regular variation in R-R intervals on ECG. Sinus arrhythmia shows a normal P wave preceding each QRS complex. The rate often increases with inspiration and decreases with expiration. It's usually seen in all 12 leads.",
            'complete right bundle branch block': "Look for a QRS duration >120 ms, rsR' pattern in V1-V3, and a wide, slurred S wave in I, V5 and V6 on the 12-lead ECG. These are indicative of a complete right bundle branch block.",
            'complete left bundle branch block': 'Look for broad QRS complex (>120ms) in leads I, V5, V6 with deep S wave in V1, V2. There should be no Q waves in lateral leads (I, aVL, V5, V6). This indicates a complete left bundle branch block.',
            'Supraventricular Tachycardia': 'Look for a rapid heart rate (>100 bpm), narrow QRS complexes (<120 ms), and absence of identifiable P waves. P waves may be hidden in preceding T wave. Focus on leads II, III, and aVF for best P wave visibility.',
            'ventricular premature complex': 'Look for premature beats with wide QRS complex (>120 ms) in the ECG. They often have a bizarre morphology, with a right bundle branch block pattern in V1 and a left bundle branch block pattern in V6. The T wave is usually in the opposite direction to the QRS complex.',
            'Left anterior fascicular block': 'Look for left axis deviation (between -45 and -90 degrees) in the frontal plane, normal or slightly widened QRS complex, and small Q waves in leads I and aVL on the 12-lead ECG to diagnose Left Anterior Fascicular Block.',
            'Atrial Flutter': 'Look for regular atrial activity at around 300 bpm in leads II, III, and aVF. The classic "sawtooth" pattern is a key feature. Ventricular rate is often around 150 bpm. Absence of P waves and presence of F waves are also indicative.',
            'Atrial Fibrillation': 'Look for irregularly irregular rhythm, absence of P waves, and variable ventricular response in the ECG. Leads II, III, and aVF often provide the clearest view of these features.',
            'atrial premature complex': 'Look for irregular rhythm in ECG. Atrial Premature Complex (APC) is characterized by early P wave, which may have different shape than normal P wave. It can be seen in any lead. The QRS complex following APC is usually normal.',
            'Normal Electrocardiogram': 'Look for regular rhythm, heart rate between 60-100 bpm, normal P wave preceding each QRS complex, PR interval 0.12-0.20 sec, and QRS duration <0.12 sec in all 12 leads. These indicate a normal ECG.',
            'Sinus Bradycardia': 'Look for a regular rhythm with a heart rate less than 60 bpm on the ECG. P waves should be present and normal, indicating the impulse is originating from the sinus node. Check all 12 leads, but lead II often provides the clearest view.',
            'Sinus Tachycardia': 'Look for a regular rhythm with a heart rate over 100 bpm in a 12-lead ECG. P waves should be present and upright in leads I, II, aVF, and V2-V6. Each P wave should be followed by a QRS complex.'}
        f = open('/home/user/tyy/project/ked/dataset/clinical_dataset/mlb12.pkl', 'rb')
        data = pickle.load(f)
        return [label_list[item] for item in data.classes_], None

    def generate_label_report(self):
        new_report = []
        for idx in range(len(self.X_data)):
            label_list = self.Y_data[idx]
            disease_label_index = np.where(label_list == 1)[0]
            background_list, label_list = [], []
            for sub_idx in disease_label_index:
                sub_label = self.label_name[sub_idx]
                label_list.append(sub_label)
                if sub_label in self.report_dict.keys():
                    background_list.append(self.report_dict[sub_label])
            background_info = ". ".join(background_list)
            diagnosis = ", ".join(label_list)
            if self.dataset_type == 'ptb-xl':
                origin_report = self.origin_report_data.iloc[idx]["target"]
                if str(origin_report) == 'nan':
                    final_report = "Background information: " +background_info
                else:
                    final_report = " This ECG is: " + origin_report + "\nBackground information: " +background_info
            else:
                final_report = " This electrocardiogram exists: " + diagnosis + "\nBackground information: " + background_info
            new_report.append(final_report)
        return new_report

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, item):
        signal = self.X_data[item]
        label = self.Y_data[item]
        if self.finetune:
            report = self.report_data[item]
        else:
            report=  ''
        return {"signal": signal, "label": label, "report": report}

class GeorgiaDataset(Dataset):
    def __init__(self, X_data, Y_data):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.label_name = self.get_label()

    def get_label(self):
        with open("/home/user/tyy/project/ked/dataset/georgia/label_list.txt", 'r') as f:
            data = f.read()
            data = data.split('\n')
        return data
    def disease_idx2name(self, idx):
        return self.label_name[idx]

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, item):
        signal = self.X_data[item]
        label = self.Y_data[item]
        return {"signal":signal, "label":label}

if __name__ == '__main__':
#     with open("/home/user/tyy/project/ked/dataset/all_label_map.json", 'r') as f:
#         all_label_map = json.load(f)
#     print(all_label_map)
#     all_label_map = {'NDT': 'non-diagnostic T wave abnormalities',
# 'NST_': 'non-specific ST segment changes',
# 'DIG': 'digitalis-effect',
# 'LNGQT': 'long QT interval',
# 'NORM': 'normal ECG',
# 'IMI': 'inferior myocardial infarction',
# 'ASMI': 'anteroseptal myocardial infarction',
# 'LVH': 'left ventricular hypertrophy',
# 'LAFB': 'left anterior fascicular block',
# 'ISC_': 'non-specific ischemic',
# 'IRBBB': 'incomplete right bundle branch block',
# '1AVB': 'first degree atrioventricular block',
# 'IVCD': 'non-specific intraventricular conduction disturbance (block)',
# 'ISCAL': 'ischemic in anterolateral leads',
# 'CRBBB': 'complete right bundle branch block',
# 'CLBBB': 'complete left bundle branch block',
# 'ILMI': 'inferolateral myocardial infarction',
# 'LAO/LAE': 'left atrial overload/enlargement',
# 'AMI': 'anterior myocardial infarction',
# 'ALMI': 'anterolateral myocardial infarction',
# 'ISCIN': 'ischemic in inferior leads',
# 'INJAS': 'subendocardial injury in anteroseptal leads',
# 'LMI': 'lateral myocardial infarction',
# 'ISCIL': 'ischemic in inferolateral leads',
# 'LPFB': 'left posterior fascicular block',
# 'ISCAS': 'ischemic in anteroseptal leads',
# 'INJAL': 'subendocardial injury in anterolateral leads',
# 'ISCLA': 'ischemic in lateral leads',
# 'RVH': 'right ventricular hypertrophy',
# 'ANEUR': 'ST-T changes compatible with ventricular aneurysm',
# 'RAO/RAE': 'right atrial overload/enlargement',
# 'EL': 'electrolytic disturbance or drug (former EDIS)',
# 'WPW': 'Wolf-Parkinson-White syndrome',
# 'ILBBB': 'incomplete left bundle branch block',
# 'IPLMI': 'inferoposterolateral myocardial infarction',
# 'ISCAN': 'ischemic in anterior leads',
# 'IPMI': 'inferoposterior myocardial infarction',
# 'SEHYP': 'septal hypertrophy',
# 'INJIN': 'subendocardial injury in inferior leads',
# 'INJLA': 'subendocardial injury in lateral leads',
# 'PMI': 'posterior myocardial infarction',
# '3AVB': 'third degree atrioventricular block',
# 'INJIL': 'subendocardial injury in inferolateral leads',
# '2AVB': 'second degree AV block',
# 'ABQRS': 'abnormal QRS',
# 'PVC': 'ventricular premature complex',
# 'STD_': 'non-specific ST segment depression',
# 'VCLVH': 'voltage criteria (QRS) for left ventricular hypertrophy',
# 'QWAVE': 'Q waves present',
# 'LOWT': 'low amplitude T wave',
# 'NT_': 'non-specific T wave changes',
# 'PAC': 'atrial premature complex',
# 'LPR': 'prolonged PR interval',
# 'INVT': 'inverted T wave',
# 'LVOLT': 'low QRS voltages in the frontal and horizontal leads',
# 'HVOLT': 'high QRS voltage',
# 'TAB_': 'T wave abnormality',
# 'STE_': 'non-specific ST segment elevation',
# 'PRC(S)': 'premature complex(es)',
# 'SR': 'sinus rhythm',
# 'AFIB': 'atrial fibrillation',
# 'STACH': 'sinus tachycardia',
# 'SARRH': 'sinus arrhythmia',
# 'SBRAD': 'sinus bradycardia',
# 'PACE': 'normal functioning artificial pacemaker',
# 'SVARR': 'supraventricular arrhythmia',
# 'BIGU': 'bigeminal pattern (unknown origin, SV or Ventricular)',
# 'AFLT': 'atrial flutter',
# 'SVTAC': 'supraventricular tachycardia',
# 'PSVT': 'paroxysmal supraventricular tachycardia',
# 'TRIGU': 'trigeminal pattern (unknown origin, SV or Ventricular)'}
#     with open("/home/user/tyy/project/ked/dataset/all_label_map_2_8.json", 'w') as f:
#         json.dump(all_label_map, f)
#     f = open('/home/user/tyy/project/ked/dataset/mimiciv/mlb.pkl', 'rb')
#     data = pickle.load(f)
#     print([item for item in data.classes_])
#     print()
    f = open('/home/user/tyy/project/ked/dataset/clinical_dataset/mlb12.pkl', 'rb')
    data = pickle.load(f)
    label_list = {
                'I°房室传导阻滞': "first degree atrioventricular block",
                'T波倒置': "inverted T wave",
                "不齐": "Sinus arrhythmia",
                "完全性右束支传导阻滞": "complete right bundle branch block",
                '完全性左束支传导阻滞': "complete left bundle branch block",
                "室上性心动过速": "Supraventricular Tachycardia",
                # "室性早搏": "ventricular premature complex:Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
                "室性早搏": "ventricular premature complex",
                "左前分支传导阻滞": "Left anterior fascicular block",
                "心房扑动": "Atrial Flutter",
                "心房颤动": "Atrial Fibrillation",
                # "房性早搏": "atrial premature complex:Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
                "房性早搏": "atrial premature complex",
                "正常心电图": "normal ECG",
                "窦性心动过缓": "Sinus Bradycardia",
                "窦性心动过速": "Sinus Tachycardia"}
    print([label_list[item] for item in data.classes_])
    print([item for item in data.classes_])
    # all_label_map = {"SB": "Sinus Bradycardia", "SR": "Sinus Rhythm", "AFIB": "Atrial Fibrillation",
    #                  "ST": "Sinus Tachycardia", "AF": "Atrial Flutter", "SA": "Sinus Arrhythmia:Focus on leads II, III, and aVF. Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
    #                  "SVT": "Supraventricular Tachycardia", "AT": "Atrial Tachycardia"}
    # # f = open('/home/user/tyy/project/ked/dataset/shaoxing/mlb.pkl', 'rb')
    # # data = pickle.load(f)
    # # all_label_map = {
    # #     "SB": "Sinus Bradycardia:Look for a regular rhythm with a heart rate below 60 bpm. The P waves should be upright and consistent, with a normal PR interval.",
    # #     "SR": "Sinus Rhythm",
    # #     "AFIB": "Atrial Fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
    # #     "ST": "Sinus Tachycardia(ST):Look for a regular rhythm, upright P waves preceding each QRS complex, and a heart rate above 100 beats per minute.",
    # #     "AF": "Atrial Flutter(AF):Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
    # #     "SA": "Sinus Arrhythmia(SA):Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
    # #     "SVT": "Supraventricular Tachycardia(SVT):Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
    # #     "AT": "Atrial Tachycardia(AT):Look for a regular narrow QRS complex rhythm with a heart rate >100 bpm. P-waves may be hidden in the preceding T-wave or may appear abnormal, often with a different morphology compared to sinus rhythm."}
    # f = open('/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1.1/data/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # print(data)
    # with open("/home/user/tyy/project/ked/dataset/shaoxing/shaoxing_label_map_report.json", 'r') as f:
    #     label_report = json.load(f)
    # print([all_label_map[item] for item in data.classes_])

    # f = open('/home/user/tyy/project/ked/dataset/georgia/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # with open("/home/user/tyy/project/ked/dataset/georgia/georgia_label_map_report.json", 'r') as f:
    #     label_report = json.load(f)
    # print([all_label_map[item] for item in data.classes_])
    # print(label_report)
    # label_name = ['Atrial fibrillation',
    #               'Atrial flutter',
    #               'first degree AV block',
    #               'Incomplete right bundle branch block',
    #               'Left axis deviation',
    #               'Left anterior fascicular block',
    #               'Left bundle branch block',
    #               'low QRS voltages in the frontal and horizontal leads',
    #               'long QT-interval',
    #               'non-specific intraventricular conduction disturbance (block)',
    #               'Sinus rhythm',
    #               'atrial premature complex',
    #               'Q wave abnormal',
    #               'Right bundle branch block',
    #               'Sinus arrhythmia',
    #               'Sinus bradycardia',
    #               'Sinus tachycardia',
    #               'T-wave abnormality',
    #               'inverted T-waves',
    #               'ventricular premature complex']
    # label_report = {'Atrial fibrillation': 'To diagnose Atrial fibrillation from a 12-lead ECG, focus on leads II, III, and aVF. Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.',
    #                 'Atrial flutter': 'To diagnose Atrial flutter on a 12-lead ECG, focus on leads II, III, and aVF. Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.',
    #                 'first degree AV block': 'To diagnose first-degree AV block on a 12-lead ECG, focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.',
    #                 'Incomplete right bundle branch block': "To diagnose Incomplete right bundle branch block from a 12-lead ECG, focus on leads V1 and V2. Look for a QRS duration of more than 120 ms, a slurred S wave in lead I, and an rSR' pattern in leads V1 and V2.",
    #                 'Left axis deviation': 'To diagnose Left Axis Deviation on a 12-lead ECG, focus on leads I and aVF. Look for a positive QRS complex in lead I and a negative QRS complex in lead aVF. If the QRS complex is positive in lead I and negative in lead aVF, it indicates Left Axis Deviation.',
    #                 'Left anterior fascicular block': 'To diagnose Left Anterior Fascicular Block on a 12-lead ECG, focus on leads I and aVL. Look for left axis deviation (> -45 degrees) and qR pattern in lead I, and rS pattern in lead aVL.',
    #                 'Left bundle branch block': 'To diagnose Left bundle branch block on a 12-lead ECG, focus on leads V1 and V6. Look for a wide QRS complex (>120 ms), broad R waves, and deep S waves in these leads. Additionally, observe the absence of Q waves in leads I, aVL, V5, and V6.',
    #                 'low QRS voltages in the frontal and horizontal leads': 'To diagnose low QRS voltages in frontal and horizontal leads from a 12-lead ECG, focus on leads I, II, III, aVR, aVL, aVF, V1, and V6. Look for QRS amplitudes less than 5 mm in limb leads and less than 10 mm in precordial leads.',
    #                 'long QT-interval': 'To diagnose long QT-interval on a 12-lead ECG, focus on leads II, V5, and V6. Look for a prolonged QT interval (>440 ms in males, >460 ms in females) and assess for T-wave abnormalities, such as T-wave notching or low-amplitude T-waves.',
    #                 'non-specific intraventricular conduction disturbance (block)': 'To diagnose non-specific intraventricular conduction disturbance (block) on a 12-lead ECG, focus on leads V1 and V6. Look for widened QRS complexes (>0.12 seconds) and abnormal QRS morphology. Absence of specific bundle branch block criteria suggests non-specific intraventricular conduction disturbance. Consult a cardiologist for further evaluation.',
    #                 'Sinus rhythm': 'To diagnose Sinus rhythm on a 12-lead ECG, focus on leads II, III, and aVF. Look for upright P waves preceding each QRS complex, with a consistent PR interval (0.12-0.20 seconds). The QRS complexes should be narrow (≤0.12 seconds) and the rate should be between 60-100 beats per minute.',
    #                 'atrial premature complex': 'To diagnose atrial premature complex (APC) on a 12-lead ECG, focus on leads II, III, and aVF. Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex. The P wave may be hidden within the preceding T wave or have a different shape.',
    #                 'Q wave abnormal': 'To diagnose Q wave abnormalities on a 12-lead ECG, focus on leads II, III, aVF, V5, and V6. Look for Q waves deeper than 1/3 the R wave height, lasting longer than 0.04 seconds, and present in at least two contiguous leads. Seek medical advice for accurate interpretation.',
    #                 'Right bundle branch block': 'To diagnose Right Bundle Branch Block (RBBB) on a 12-lead ECG, focus on leads V1 and V2. Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in these leads. Additionally, the presence of a secondary R wave in lead V1 may indicate RBBB.',
    #                 'Sinus arrhythmia': 'To diagnose Sinus arrhythmia from a 12-lead ECG, focus on leads II, III, and aVF. Look for irregular R-R intervals that vary with respiration. The P-wave morphology should remain consistent.',
    #                 'Sinus bradycardia': 'To diagnose Sinus bradycardia on a 12-lead ECG, focus on leads II, III, and aVF. Look for a regular rhythm with a heart rate <60 bpm. P-waves should be present and upright in leads II and aVF. QRS complexes should be normal.',
    #                 'Sinus tachycardia': 'To diagnose sinus tachycardia on a 12-lead ECG, focus on leads II, III, and aVF. Look for a regular rhythm with a heart rate >100 bpm. P-waves should be upright in leads II and aVF, and the PR interval should be normal.',
    #                 'T-wave abnormality': 'To diagnose T-wave abnormalities on a 12-lead ECG, focus on leads V2-V6. Look for inverted or peaked T-waves, asymmetry, or changes in amplitude. Compare with other leads for confirmation. Seek medical advice for accurate interpretation and diagnosis.',
    #                 'inverted T-waves': 'To diagnose inverted T-waves on a 12-lead ECG, focus on leads V1-V6 and the precordial leads. Look for symmetrically inverted T-waves, especially in leads with positive QRS complexes. Consider underlying causes like myocardial ischemia, ventricular hypertrophy, or electrolyte imbalances. Seek expert consultation for accurate interpretation.',
    #                 'ventricular premature complex': 'To diagnose ventricular premature complex (VPC) from a 12-lead ECG, focus on leads V1 and V6. Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may also have a different axis or bundle branch block pattern.'}
