# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-15 20:22
import pandas as pd
import json
import requests

error_list = [1]
file_path = "res_report_train_clean.csv"

def clean_data():
    report_file = pd.read_csv(file_path, index_col=[0])
    error_file = report_file.iloc[error_list]
    error_data = error_file["source"]
    for idx, item in error_data.items():
        result = handler_request(item)
        report_file.iloc[idx, 1] = result
    report_file.to_csv("res_report_train_clean.csv", index=True, header=True)

error_count = 0
def handler_request(item):
    prompt_prefix_diagnosis = "Help me translate the medical report from German into English. Please directly tell me the translation result, no other explanatory words. The origin medical report is: "
    url = "XXXX"    # your chatgpt query url
    headers = {"Content-Type": "application/json;charset=utf-8",
               "Accept": "*/*",
               "Accept-Encoding": "gzip, deflate, br",
               "Connection": "keep-alive"}
    try:
        data = {"messages": [
            {"role": "user", "content": prompt_prefix_diagnosis + item}],
            "userId": "serveForPaper"}
        json_data = json.dumps(data)
        response = requests.post(url=url, data=json_data, headers=headers)
        json_response = response.json()
        return json_response["content"].replace("\n\t", "").replace("\n", "")
    except Exception as e:
        print(e)
        global error_count
        error_count += 1
        print(error_count)
        return "error"

def clean_data__(csv_file=None):
    # case1:"(The report indicates that the patient has a normal heart rhythm and a normal electrocardiogram)"
    # case2: "Translation:"
    # case3: Less than four words to translate
    # I'm sorry, but I cannot provide a translation without the actual medical report in German. Please provide the text that needs to be translated.
    if csv_file is not None:
        report_file = csv_file
    else:
        report_file = pd.read_csv("report_train_clean.csv", index_col=[0])
    case3_list = []
    for idx, item in report_file.iterrows():
        # print(item)
        data = item['target']
        if len(data.split()) < 4:
            case3_list.append(idx)
    print(len(case3_list))
    error_file = report_file.loc[case3_list]
    error_data = error_file["source"]
    for idx, item in error_data.items():
        result = handler_request(item)
        report_file.loc[idx, "target"] = result

    for idx, item in report_file.iterrows():
        target = item['target']
        source = item['source']
        if "(The report indicates that the patient" in target:
            new_data = target.split("(The report indicates that the patient")[0]
            report_file.loc[idx, "target"] = new_data
        if "Translation: " in target:
            new_data = target.split("Translation: ")
            if not new_data[0]:
                report_file.loc[idx, "target"] = new_data[1]
            elif len(source.split()) >= 2 * len(new_data[0].split()):
                report_file.loc[idx, "target"] = new_data[1]
            else:
                report_file.loc[idx, "target"] = new_data[0]
        if "I'm sorry" in target:
            report_file.loc[idx, "target"] = ""
        # if "Note:" in target:
        #     if "Edit" in source:
        #         continue
        #     new_data = target.split("Note:")
        #     report_file.iloc[idx, 1] = new_data[0]
        if len(source.split()) >= 2 * len(target.split()):
            result = handler_request(source)
            if "Translation: " in target:
                new_data = target.split("Translation: ")
                if not new_data[0]:
                    report_file.loc[idx, "target"] = new_data[1]
                elif len(source.split()) >= 2 * len(new_data[0].split()):
                    report_file.loc[idx, "target"] = new_data[1]
                else:
                    report_file.loc[idx, "target"] = new_data[0]
            else:
                report_file.loc[idx, "target"] = result
        # if "already in English" in target:
        #     report_file.iloc[idx, 1] = source

    report_file.to_csv("res_report_train_clean_final.csv", index=True, header=True)

from codes.utils.utils import translate_report

def res_data_trans():
    file_path = 'res_labels.csv'
    res_report = pd.read_csv(file_path, index_col=[0])
    res_report = res_report['report']
    translate_report(res_report.values, '/home/tyy/ecg_ptbxl/output/exp1.1/data/res_report_train_clean.csv')

def gene_total_report():
    index = pd.read_csv('diagnosis_index_temp.csv', index_col=[0])
    report1 = pd.read_csv('report_train_clean_final.csv', index_col=[0])
    report1.index = index.index
    report2 = pd.read_csv('res_report_train_clean_final.csv', index_col=[0])
    merged_df = pd.concat([report1, report2], axis=0, sort=False)
    merged_df1 = merged_df.sort_index()
    merged_df1.to_csv("total_report_train_final.csv", index=True)
    print(merged_df.head())

def gene_form_report():
    index_form = pd.read_csv('../../exp2/data/report_index_temp.csv', index_col=[0])
    total_report = pd.read_csv('total_report_train_final.csv', index_col=[0])
    form_report = total_report.loc[index_form.index]
    form_report.to_csv("../../exp2/data/total_report_train_final.csv", index=True)

    index_rhythm = pd.read_csv('../../exp3/data/report_index_temp.csv', index_col=[0])
    rhythm_report = total_report.loc[index_rhythm.index]
    rhythm_report.to_csv("../../exp3/data/total_report_train_final.csv", index=True)



if __name__ == '__main__':
    # clean_data()
    # csv1 = pd.read_csv('res_labels.csv', index_col=[0])
    # csv2 = pd.read_csv('res_report_train_clean.csv', index_col=None)
    # csv2.index = csv1.index
    # csv2 = csv2[['source', 'target']]
    # clean_data__(csv2)
    gene_form_report()

