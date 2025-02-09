# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2024-01-29 15:26

import json
import requests
import pandas as pd
from sklearn.preprocessing import StandardScaler, MultiLabelBinarizer
import pickle
import psycopg2
conn = psycopg2.connect(host="192.168.40.125",
                        database="postgres",
                        user="postgres",
                        password="tyy123",
                        port="5432")

"""
Refer to this project： https://github.com/MIT-LCP/mimic-code to construct the mimic-iv database 
"""

def generate_ked_label():
    # annotated_data = pd.read_excel('mimicivecg_labeling_final_anno.xlsx', sheet_name='mimicecg_label_round2')
    # annotated_data = annotated_data[['subperClass', 'label', 'count', 'final']]
    # annotated_dict = annotated_data.set_index('label')['final'].to_dict()
    # total_label_set = set()
    # for key, value in annotated_dict.items():
    #     if '#' in value:
    #         total_label_set.update(value.split('#'))
    #     else:
    #         total_label_set.add(value)
    # total_label_set.remove("delete")
    # total_label_set.remove('delete_all')
    # print(len(total_label_set))  # 345
    # new_df = pd.DataFrame({"total_label": list(total_label_set)})
    # new_df.to_csv("./new_process_01_29/total_label_set.csv")
    query_1 = """SELECT subject_id,study_id,report_0,report_1,report_2,report_3,report_4,
                         report_5,report_6,report_7, report_8, report_9,report_10,
                         report_11,report_12,report_13, report_14, report_15,report_16,
                         report_17 FROM mimiciv_ecg.machine_measurements"""
    # load the tables into dataframes
    df_1 = pd.read_sql(query_1, conn)
    def combine_reports(row):
        reports = [str(elem) for elem in row if elem]  # 确保空值被忽略
        return ', '.join(reports)

    report_cols = [col for col in df_1.columns if 'report_' in col]
    df_1['report'] = df_1[report_cols].apply(combine_reports, axis=1)
    df_1['study_id'] = df_1['study_id'].astype(str)
    df_1['subject_id'] = df_1['subject_id'].astype(str)
    df_1 = df_1[['subject_id', "study_id", "report"]]

    label_data = pd.read_json(
        "/home/user/tyy/project/imagebind/datasets/mimiciv-ecg/mimiciv_ecg_label_annotated_11_9.json")
    label_data['study_id'] = label_data['study_id'].astype(str)
    label_data['subject_id'] = label_data['subject_id'].astype(str)

    label_set_count = {}
    for idx, item in label_data.iterrows():
        for label in item['labels']:
            if label not in label_set_count.keys():
                label_set_count[label] = 1
            else:
                label_set_count[label] += 1
    label_set_count = {k: v for k, v in sorted(label_set_count.items(), key=lambda item: item[1], reverse=True)}
    sorted_filter_dict = {key: value / label_data.shape[0] for key, value in label_set_count.items() if value >= 2000}
    label_data['filter_labels'] = label_data['labels'].apply(lambda x: [i for i in x if i in sorted_filter_dict.keys()])
    label_data = label_data[label_data['filter_labels'].map(lambda d: len(d)) > 0]

    # label_data = label_data.set_index('study_id')
    print(label_data.shape)
    label_data = pd.merge(label_data, df_1, on=['subject_id', 'study_id'], how='inner')
    print(label_data.shape)
    print(label_data['study_id'].nunique())
    print(label_data['subject_id'].nunique())

    mlb = MultiLabelBinarizer()
    mlb.fit(label_data.filter_labels.values)
    y = mlb.transform(label_data.filter_labels.values)
    print(y.shape)
    total_patient_id = list(set(label_data['subject_id'].values.tolist()))
    patient_id_list_train = total_patient_id[:int(len(total_patient_id) * 0.9)]
    patient_id_list_val = total_patient_id[int(len(total_patient_id) * 0.9):int(len(total_patient_id) * 0.95)]
    patient_id_list_test = total_patient_id[int(len(total_patient_id) * 0.95):]
    with open("patient_id_list_01_29.json", "w") as f:
        json.dump({"train_list": patient_id_list_train, "val_list": patient_id_list_val,
                   "test_list": patient_id_list_test}, f)
    data_y_total_train = label_data[label_data['subject_id'].isin(patient_id_list_train)]
    data_y_total_val = label_data[label_data['subject_id'].isin(patient_id_list_val)]
    data_y_total_test = label_data[label_data['subject_id'].isin(patient_id_list_test)]
    y_train = y[label_data['subject_id'].isin(patient_id_list_train)]
    y_val = y[label_data['subject_id'].isin(patient_id_list_val)]
    y_test = y[label_data['subject_id'].isin(patient_id_list_test)]
    print(data_y_total_train.shape)
    print(data_y_total_val.shape)
    print(data_y_total_test.shape)
    print(y_train.shape)
    print(y_val.shape)
    print(y_test.shape)
    print()

    y_train.dump('y_train_one_hot_data.npy')
    y_val.dump('y_val_one_hot_data.npy')
    y_test.dump('y_test_one_hot_data.npy')

    data_y_total_train.to_json('data_y_total_train.json', orient='records')
    data_y_total_val.to_json('data_y_total_val.json', orient='records')
    data_y_total_test.to_json('data_y_total_test.json', orient='records')

    with open('mlb.pkl', 'wb') as tokenizer:
        pickle.dump(mlb, tokenizer)


def generate_label_description():
    """"""
    label_set_df = pd.read_csv('total_label_set.csv')
    label_set_df = label_set_df[~label_set_df['total_label'].isin(['delete', 'delete_all'])]
    total_label_list = label_set_df['total_label'].values.tolist()
    with open('descript.txt', 'r') as file:
        list_output = file.readlines()
    description = [line.strip() for line in list_output]
    generated_description_dict = {}
    for idx, item in enumerate(description):
        generated_description_dict[total_label_list[idx]] = item

    for item in total_label_list:
        if item in generated_description_dict.keys():
            continue
        generated_description_dict[item] = _handler_generate_augment_(item)

    with open("mimiciv_label_map_report.json", "w") as f:
        json.dump(generated_description_dict, f)


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


"""2024年2月27日"""
def generate_zhipuai_augment():
    """"""
    from zhipuai import ZhipuAI

    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                                  "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."

    client = ZhipuAI(api_key="YOUR_ZHIPU_API_KEY")  # 填写您自己的APIKey

    label_set_df = pd.read_csv('total_label_set.csv')
    label_set_df = label_set_df[~label_set_df['total_label'].isin(['delete', 'delete_all'])]
    total_label_list = label_set_df['total_label'].values.tolist()
    generated_description_dict = {}

    for item in total_label_list:
        if item in generated_description_dict.keys():
            continue
        response = client.chat.completions.create(
            model="glm-4",  # 填写需要调用的模型名称
            messages=[
                {"role": "user", "content": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis},
            ],
        )
        response = response.choices[0].message.content
        print(response)
        generated_description_dict[item] = response

    with open("mimiciv_label_map_report_zhipuai.json", "w") as f:
        json.dump(generated_description_dict, f)

    print()

def refine_zhipu_augment():
    from zhipuai import ZhipuAI

    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                              "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                              "than 50 words."
    client = ZhipuAI(api_key="YOUR_ZHIPU_API_KEY")  # 填写您自己的APIKey
    def contains_chinese(s):
        return any('\u4e00' <= c <= '\u9fff' for c in s)
    with open("mimiciv_label_map_report_zhipuai_new.json", "r") as f:
        generated_description_dict = json.load(f)
    new_description_dict = {}
    for key, value in generated_description_dict.items():
        if contains_chinese(value):
            response = client.chat.completions.create(
                model="glm-4",  # 填写需要调用的模型名称
                messages=[
                    {"role": "user", "content": prompt_prefix_diagnosis + key + prompt_suffix_diagnosis},
                ],
            )
            response = response.choices[0].message.content
            print(key, response)
            new_description_dict[key] = response
        else:
            new_description_dict[key] = value
    with open("mimiciv_label_map_report_zhipuai_new.json", "w") as f:
        json.dump(new_description_dict, f)

def generate_gemini_augment():
    label_set_df = pd.read_csv('total_label_set.csv')
    label_set_df = label_set_df[~label_set_df['total_label'].isin(['delete', 'delete_all'])]
    total_label_list = label_set_df['total_label'].values.tolist()
    generated_description_dict = {}

    for item in total_label_list:
        response = _generate_gemini_augment_(item)
        print(response)
        generated_description_dict[item] = response

    with open("mimiciv_label_map_report_gemini.json", "w") as f:
        json.dump(generated_description_dict, f)

def _generate_gemini_augment_(item):
    prompt_prefix_diagnosis = "I want you to play the role of a professional Electrocardiologist, and I need you to teach me how " \
                              "to diagnose "
    prompt_suffix_diagnosis = " from 12-lead ECG. such as what leads or what features to focus on ,etc. Your answer must be less " \
                                  "than 50 words."
    url = "YOUR_GEMINI_API"
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    data = {"messages": prompt_prefix_diagnosis + item + prompt_suffix_diagnosis,
            "userId": "serveForPaper"}
    json_data = json.dumps(data)
    response = requests.post(url=url, data=json_data, headers=headers)
    return response.text

# 2024年9月6日
def generate_age_sex():
    """"""
    query_1 = """SELECT subject_id,study_id,report_0,report_1,report_2,report_3,report_4,
                         report_5,report_6,report_7, report_8, report_9,report_10,
                         report_11,report_12,report_13, report_14, report_15,report_16,
                         report_17 FROM mimiciv_ecg.machine_measurements"""

    query_2 = """SELECT subject_id, study_id, ecg_time, path FROM mimiciv_ecg.record_list"""

    # load the tables into dataframes
    df_1 = pd.read_sql(query_1, conn)
    df_2 = pd.read_sql(query_2, conn)
    df_2['ecg_time'] = pd.to_datetime(df_2['ecg_time'])

    # merge the dataframes on the columns ['subject_id', 'study_id']
    merged_df = pd.merge(df_1, df_2, on=['subject_id', 'study_id'])

    query_3 = """SELECT subject_id,admittime, dischtime,age,gender, hospital_mortality, one_year_mortality FROM mimiciv_derived.hosp_demographics_new"""

    # 读取第一个表的数据
    df3 = pd.read_sql(query_3, conn)

    # 确保你的时间字段是datetime类型
    df3['admittime'] = pd.to_datetime(df3['admittime'])
    df3['dischtime'] = pd.to_datetime(df3['dischtime'])
    df3['subject_id'] = df3['subject_id'].astype(str)

    final_merged_df = pd.merge(merged_df, df3, on='subject_id')
    print(final_merged_df.shape)
    cond = (final_merged_df['ecg_time'] >= final_merged_df['admittime']) & (final_merged_df['ecg_time'] <= final_merged_df['dischtime'])
    final_merged_df = final_merged_df[cond]
    print(final_merged_df['study_id'].nunique())
    print(final_merged_df['subject_id'].nunique())
    df = final_merged_df
    # 统计年龄的最小最大范围（忽略NaN）
    最小年龄 = df['age'].min(skipna=True)
    最大年龄 = df['age'].max(skipna=True)
    print(f"年龄的范围是从 {最小年龄} 到 {最大年龄}")

    # 计算年龄的均值和方差（忽略NaN）
    均值 = df['age'].mean(skipna=True)
    方差 = df['age'].std(skipna=True)
    print(f"年龄的均值是 {均值}，方差是 {方差}")

    # 计算男性的比例
    男性数量 = len(df[df['gender'] == 'M'])
    总人数 = len(df)
    男性比例 = 男性数量 / 总人数
    print(f"男性的比例是 {男性比例}")

    # 确保输出结果都能被显示
    print(f"最小年龄: {最小年龄}, 最大年龄: {最大年龄}")
    print(f"均值: {均值}, 方差: {方差}")
    print(f"男性比例: {男性比例}")
    print(final_merged_df)

if __name__ == '__main__':
    # generate_age_sex()
    # generate_gemini_augment()

    # refine_zhipu_augment()

    # generate_zhipuai_augment()

    generate_ked_label()
    # generate_label_description()
