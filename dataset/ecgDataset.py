# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-09 17:13
import torch
from torch.utils.data import Dataset
import numpy as np

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
    def __init__(self, X_data, Y_data, report=None, useAugment=True, use_what_label='diagnosis_label', use_what_prompt='base'):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.background_info, self.all_label_map = self.get_background_infp(use_what_label, use_what_prompt)
        self.label_list = self.get_label_list(use_what_label)
        if report is not None:
            if useAugment:
                self.report_data = self.report_augment(report["target"].values)
            else:
                self.report_data = [item if not str(item) == 'nan' else "" for item in report["target"].values]
        else:
            self.report_data = None


    def __len__(self):
        return len(self.X_data)

    def get_label_list(self, use_what_label):
        if use_what_label == 'diagnosis_label':
            f = open('/home/tyy/ecg_ptbxl/output/exp1/data/mlb.pkl', 'rb')
        elif use_what_label == 'all':
            f = open('/home/tyy/ecg_ptbxl/output/exp0/data/mlb.pkl', 'rb')
        elif use_what_label == 'subdiagnosis_label':
            f = open('/home/tyy/ecg_ptbxl/output/exp1.1/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        return data.classes_

    def get_background_infp(self, use_what_label, use_what_prompt="base"):
        if use_what_label in ["diagnosis_label", "all"]:
            if use_what_prompt == 'base':
                background_path = "/home/tyy/unECG/dataset/all_label_augment.json"
            elif use_what_prompt == 'concise':
                background_path = "/home/tyy/unECG/dataset/prompt_label/label_map_concise.json"
            elif use_what_prompt == 'plain_diagnosis':
                background_path = "/home/tyy/unECG/dataset/prompt_label/label_map_plain_diagnosis.json"
            elif use_what_prompt == 'intern':
                background_path = "/home/tyy/unECG/dataset/prompt_label/label_map_intern.json"
            with open(background_path, 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
                all_label_map = json.load(f)

            return background_info, all_label_map
        elif use_what_label == "subdiagnosis_label":
            with open("/home/tyy/unECG/dataset/label_augment_23.json", 'r') as f:
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
    def __init__(self, X_data, Y_data, dataset_type):
        self.X_data = torch.FloatTensor(X_data.astype('float64'))
        self.Y_data = Y_data
        self.dataset_type = dataset_type
        if self.dataset_type == 'georgia':
            self.label_name, self.report_dict = self.load_georgia_label_and_report()
        elif self.dataset_type == 'shaoxing':
            self.label_name, self.report_dict = self.load_shaoxing_label_and_report()
        elif self.dataset_type == 'cpsc':
            self.label_name, self.report_dict = self.load_cpsc_label_and_report()
        elif self.dataset_type == 'clinical':
            self.label_name, self.report_dict = self.load_clinical_label_and_report()
        if self.dataset_type not in ['clinical']:
            self.report_data = self.generate_label_report()

    def load_georgia_label_and_report(self):
        with open("/home/tyy/unECG/dataset/georgia/label_map.json", 'r') as f:
            all_label_map = json.load(f)
        f = open('/home/tyy/unECG/dataset/georgia/mlb.pkl', 'rb')
        data = pickle.load(f)
        with open("/home/tyy/unECG/dataset/georgia/georgia_label_map_report.json", 'r') as f:
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
        # label_report = {
        #     'Atrial fibrillation': 'To diagnose Atrial fibrillation from a 12-lead ECG, focus on leads II, III, and aVF. Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.',
        #     'Atrial flutter': 'To diagnose Atrial flutter on a 12-lead ECG, focus on leads II, III, and aVF. Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.',
        #     'first degree AV block': 'To diagnose first-degree AV block on a 12-lead ECG, focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.',
        #     'Incomplete right bundle branch block': "To diagnose Incomplete right bundle branch block from a 12-lead ECG, focus on leads V1 and V2. Look for a QRS duration of more than 120 ms, a slurred S wave in lead I, and an rSR' pattern in leads V1 and V2.",
        #     'Left axis deviation': 'To diagnose Left Axis Deviation on a 12-lead ECG, focus on leads I and aVF. Look for a positive QRS complex in lead I and a negative QRS complex in lead aVF. If the QRS complex is positive in lead I and negative in lead aVF, it indicates Left Axis Deviation.',
        #     'Left anterior fascicular block': 'To diagnose Left Anterior Fascicular Block on a 12-lead ECG, focus on leads I and aVL. Look for left axis deviation (> -45 degrees) and qR pattern in lead I, and rS pattern in lead aVL.',
        #     'Left bundle branch block': 'To diagnose Left bundle branch block on a 12-lead ECG, focus on leads V1 and V6. Look for a wide QRS complex (>120 ms), broad R waves, and deep S waves in these leads. Additionally, observe the absence of Q waves in leads I, aVL, V5, and V6.',
        #     'low QRS voltages in the frontal and horizontal leads': 'To diagnose low QRS voltages in frontal and horizontal leads from a 12-lead ECG, focus on leads I, II, III, aVR, aVL, aVF, V1, and V6. Look for QRS amplitudes less than 5 mm in limb leads and less than 10 mm in precordial leads.',
        #     'long QT-interval': 'To diagnose long QT-interval on a 12-lead ECG, focus on leads II, V5, and V6. Look for a prolonged QT interval (>440 ms in males, >460 ms in females) and assess for T-wave abnormalities, such as T-wave notching or low-amplitude T-waves.',
        #     'non-specific intraventricular conduction disturbance (block)': 'To diagnose non-specific intraventricular conduction disturbance (block) on a 12-lead ECG, focus on leads V1 and V6. Look for widened QRS complexes (>0.12 seconds) and abnormal QRS morphology. Absence of specific bundle branch block criteria suggests non-specific intraventricular conduction disturbance. Consult a cardiologist for further evaluation.',
        #     'Sinus rhythm': 'To diagnose Sinus rhythm on a 12-lead ECG, focus on leads II, III, and aVF. Look for upright P waves preceding each QRS complex, with a consistent PR interval (0.12-0.20 seconds). The QRS complexes should be narrow (≤0.12 seconds) and the rate should be between 60-100 beats per minute.',
        #     'atrial premature complex': 'To diagnose atrial premature complex (APC) on a 12-lead ECG, focus on leads II, III, and aVF. Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex. The P wave may be hidden within the preceding T wave or have a different shape.',
        #     'Q wave abnormal': 'To diagnose Q wave abnormalities on a 12-lead ECG, focus on leads I, aVL, V5, and V6. Look for Q waves that are deeper than 25% of the corresponding R wave amplitude and wider than 0.04 seconds. A Q wave in leads V1 and V2 wider than 0.04 seconds may indicate an anterior myocardial infarction.',
        #     'Right bundle branch block': 'To diagnose Right Bundle Branch Block (RBBB) on a 12-lead ECG, focus on leads V1 and V2. Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in these leads. Additionally, the presence of a secondary R wave in lead V1 may indicate RBBB.',
        #     'Sinus arrhythmia': 'To diagnose Sinus arrhythmia from a 12-lead ECG, focus on leads II, III, and aVF. Look for irregular R-R intervals that vary with respiration. The P-wave morphology should remain consistent.',
        #     'Sinus bradycardia': 'To diagnose Sinus bradycardia on a 12-lead ECG, focus on leads II, III, and aVF. Look for a regular rhythm with a heart rate <60 bpm. P-waves should be present and upright in leads II and aVF. QRS complexes should be normal.',
        #     'Sinus tachycardia': 'To diagnose sinus tachycardia on a 12-lead ECG, focus on leads II, III, and aVF. Look for a regular rhythm with a heart rate >100 bpm. P-waves should be upright in leads II and aVF, and the PR interval should be normal.',
        #     'T-wave abnormality': 'To diagnose T-wave abnormalities on a 12-lead ECG, focus on leads V2-V6. Look for inverted or peaked T-waves, asymmetry, or changes in amplitude. Compare with other leads for confirmation. Seek medical advice for accurate interpretation and diagnosis.',
        #     'inverted T-waves': 'To diagnose inverted T-waves on a 12-lead ECG, focus on leads V1-V6 and the precordial leads. Look for symmetrically inverted T-waves, especially in leads with positive QRS complexes. Consider underlying causes like myocardial ischemia, ventricular hypertrophy, or electrolyte imbalances. Seek expert consultation for accurate interpretation.',
        #     'ventricular premature complex': 'To diagnose ventricular premature complex (VPC) from a 12-lead ECG, focus on leads V1 and V6. Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may also have a different axis or bundle branch block pattern.'}
        return label_name, label_report

    def load_shaoxing_label_and_report(self):
        # all_label_map = {"SB": "Sinus Bradycardia", "SR": "Sinus Rhythm", "AFIB": "Atrial Fibrillation",
        #                  "ST": "Sinus Tachycardia", "AF": "Atrial Flutter", "SA": "Sinus Arrhythmia",
        #                  "SVT": "Supraventricular Tachycardia", "AT": "Atrial Tachycardia"}
        all_label_map = {"SB": "Sinus Bradycardia:Look for a regular rhythm with a heart rate below 60 bpm. The P waves should be upright and consistent, with a normal PR interval.",
                         "SR": "Sinus Rhythm",
                         "AFIB": "Atrial Fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
                         "ST": "Sinus Tachycardia(ST):Look for a regular rhythm, upright P waves preceding each QRS complex, and a heart rate above 100 beats per minute.",
                         "AF": "Atrial Flutter(AF):Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
                         "SA": "Sinus Arrhythmia(SA):Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
                         "SVT": "Supraventricular Tachycardia(SVT):Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
                         "AT": "Atrial Tachycardia(AT):Look for a regular narrow QRS complex rhythm with a heart rate >100 bpm. P-waves may be hidden in the preceding T-wave or may appear abnormal, often with a different morphology compared to sinus rhythm."}
        f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
        data = pickle.load(f)
        with open("/home/tyy/unECG/dataset/shaoxing/shaoxing_label_map_report.json", 'r') as f:
            label_report = json.load(f)
        return [all_label_map[item] for item in data.classes_], label_report

    def load_cpsc_label_and_report(self):
        # label_list = ['normal ECG', 'Atrial fibrillation', 'first degree AV block', 'left bundle branch block','Right bundle branch block',
        #                         "atrial premature complex","ventricular premature complex",'non-specific ST depression','non-specific ST elevation']
        label_list = ['normal ECG',
                      'Atrial fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.',
                      'first degree AV block(IAVB):focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.',
                      'left bundle branch block(LBBB):Look for a wide QRS complex (>120 ms), broad R waves, and deep S waves in these leads. Additionally, observe the absence of Q waves in leads I, aVL, V5, and V6.',
                      "Right bundle branch block(RBBB):Look for a wide QRS complex (>0.12 seconds) with an rSR' pattern in these leads. The presence of a slurred S wave in leads I, aVL, and V6 can also indicate RBBB.",
                      "Atrial premature complex(APC):Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
                      "Ventricular premature complex(VPC):Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
                      'non-specific ST depression(STD_):Look for downsloping or horizontal ST segment depression ≥1mm below the isoelectric line, often accompanied by T wave inversion.',
                      'non-specific ST elevation(STE_):Look for an elevation of the ST segment above the baseline of at least 1mm in two consecutive leads.']
        # label_list = ['normal ECG',
        #               'Atrial fibrillation(AFIB)',
        #               'first degree AV block(IAVB)',
        #               'left bundle branch block(LBBB)',
        #               "Right bundle branch block(RBBB)",
        #               "Atrial premature complex(APC)",
        #               "Ventricular premature complex(VPC)",
        #               'ST segment depression',
        #               'ST segment elevation']

        with open("/home/tyy/unECG/dataset/cpsc_label_map_report.json", 'r') as f:
            label_report = json.load(f)
        return label_list, label_report

    def load_clinical_label_and_report(self):
        # label_list = ['I°房室传导阻滞' 'T波倒置' '不齐' '完全性右束支传导阻滞' '完全性左束支传导阻滞' '室上性心动过速' '室性早搏'
        #                 '左前分支传导阻滞' '心房扑动' '心房颤动' '房性早搏' '正常心电图' '窦性心动过缓' '窦性心动过速']
        # label_list = {'I°房室传导阻滞':"first degree AV block:focus on the PR interval duration. It will be prolonged (>200 ms) and consistent across all leads.",
        #               'T波倒置':"inverted T-waves",
        #               "不齐":"Sinus arrhythmia:Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
        #               "完全性右束支传导阻滞":"complete right bundle branch block:Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in these leads. Additionally, the presence of a secondary R wave in lead V1 may indicate RBBB.",
        #               '完全性左束支传导阻滞':"complete left bundle branch block",
        #               "室上性心动过速":"Supraventricular Tachycardia:Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
        #               "室性早搏":"ventricular premature complex:Look for wide QRS complexes (>0.12 seconds) with abnormal morphology, absence of preceding P waves, and compensatory pause. VPCs may have different shapes, so careful analysis is crucial.",
        #               "左前分支传导阻滞":"Left anterior fascicular block:Look for left axis deviation (> -45 degrees) and qR pattern in lead I, and rS pattern in lead aVL.",
        #               "心房扑动":"Atrial Flutter:Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
        #               "心房颤动":"Atrial Fibrillation:Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
        #               "房性早搏":"atrial premature complex:Look for an abnormal P wave morphology, occurring earlier than expected, followed by a premature QRS complex.",
        #               "正常心电图":"Normal ECG",
        #               "窦性心动过缓":"Sinus Bradycardia",
        #               "窦性心动过速":"Sinus Tachycardia"}
        label_list = {
            'I°房室传导阻滞': "first degree AV block",
            'T波倒置': "inverted T-waves",
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
            "正常心电图": "Normal Electrocardiogram",
            "窦性心动过缓": "Sinus Bradycardia",
            "窦性心动过速": "Sinus Tachycardia"}
        f = open('/home/tyy/unECG/dataset/clinical_dataset/mlb12.pkl', 'rb')
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
            final_report = "Background information: " + background_info + " This electrocardiogram exists: " + diagnosis
            new_report.append(final_report)
        return new_report

    def __len__(self):
        return len(self.X_data)

    def __getitem__(self, item):
        signal = self.X_data[item]
        label = self.Y_data[item]
        if self.dataset_type not in ['clinical']:
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
        with open("/home/tyy/unECG/dataset/georgia/label_list.txt", 'r') as f:
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
    all_label_map = {"SB": "Sinus Bradycardia", "SR": "Sinus Rhythm", "AFIB": "Atrial Fibrillation",
                     "ST": "Sinus Tachycardia", "AF": "Atrial Flutter", "SA": "Sinus Arrhythmia:Focus on leads II, III, and aVF. Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
                     "SVT": "Supraventricular Tachycardia", "AT": "Atrial Tachycardia"}
    # f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # all_label_map = {
    #     "SB": "Sinus Bradycardia:Look for a regular rhythm with a heart rate below 60 bpm. The P waves should be upright and consistent, with a normal PR interval.",
    #     "SR": "Sinus Rhythm",
    #     "AFIB": "Atrial Fibrillation(AFIB):Look for an irregularly irregular rhythm, absence of P waves, and fibrillatory waves.",
    #     "ST": "Sinus Tachycardia(ST):Look for a regular rhythm, upright P waves preceding each QRS complex, and a heart rate above 100 beats per minute.",
    #     "AF": "Atrial Flutter(AF):Look for a sawtooth pattern with a regular atrial rate of 250-350 bpm. The ventricular rate is usually regular but can vary.",
    #     "SA": "Sinus Arrhythmia(SA):Look for irregular R-R intervals that vary with respiration. The heart rate should increase during inspiration and decrease during expiration.",
    #     "SVT": "Supraventricular Tachycardia(SVT):Look for a narrow QRS complex, absent P waves, and a regular rhythm. If present, a retrograde P wave after the QRS complex suggests SVT. Seek medical consultation for accurate diagnosis and treatment.",
    #     "AT": "Atrial Tachycardia(AT):Look for a regular narrow QRS complex rhythm with a heart rate >100 bpm. P-waves may be hidden in the preceding T-wave or may appear abnormal, often with a different morphology compared to sinus rhythm."}
    f = open('/home/tyy/unECG/dataset/shaoxing/mlb.pkl', 'rb')
    data = pickle.load(f)
    with open("/home/tyy/unECG/dataset/shaoxing/shaoxing_label_map_report.json", 'r') as f:
        label_report = json.load(f)
    print([all_label_map[item] for item in data.classes_])

    # f = open('/home/tyy/unECG/dataset/georgia/mlb.pkl', 'rb')
    # data = pickle.load(f)
    # with open("/home/tyy/unECG/dataset/georgia/georgia_label_map_report.json", 'r') as f:
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
