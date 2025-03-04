# ECGFM-KED

**2025-02-24** We have updated the model weight file best_valid_all_increase_with_augment_epoch_3.pt (trained on mimic-iv-ecg) and the MIMIC-IV-ECG labels mimiciv_ecg_label_annotated_11_9.json, which were annotated and cleaned by our team, on Zenodo (https://zenodo.org/records/14881564).

This repository is accompanying our article **Foundation model of ECG diagnosis: Diagnostics and explanations of any form and rhythm on ECG**. The paper has been accepted by Cell Reports Medicine (https://doi.org/10.1016/j.xcrm.2024.101875)
        
        
        
        .

To install Python dependencies:

```
# python version is 3.8.18
pip install -r requirements.txt
```

## Data
### MIMIC-IV
1. Download raw data from https://physionet.org/content/mimic-iv-ecg/1.0/ and store it in./dataset/mimiciv/
2. Modify and run the python file: ./dataset/mimiciv/data_process.py 

### PTB-XL
1. Download raw data from https://physionet.org/content/ptb-xl/1.0.1/ and store it in./dataset/ptb-xl/
2. Modify and run the python file: ./dataset/ptb-xl/data_preprocess.py 

### georgia
1. Download raw data from https://moody-challenge.physionet.org/2020/ and store it in./dataset/georgia/
2. Modify and run the python file: ./dataset/georgia/data_preprocess.py

### shaoxing(Chapman)
1. Download raw data from https://doi.org/10.6084/m9.figshare.c.4560497.v2
        
        
        
         and store it in./dataset/shaoxing/
2. Modify and run the python file: ./dataset/shaoxing/data_preprocess.py

### cpsc2018
1. Download raw data from http://2018.icbeb.org/Challenge.html and store it in./dataset/cpsc/
2. Modify and run the python file: ./dataset/cpsc/data_preprocess.py

### clinical_dataset

Download the data from https://drive.google.com/file/d/1d2GnUm2S9s9ExrkOnG4xSBD-RGLGETna/view?usp=sharing and unzip it into ./dataset/clinical_dataset/

## Pre-training

Change the configs/Res_train.yaml file to fit your needs, and change the path to the data file in main_mimiciv.py to be where you store your data, and then run main.py
## Zero-shot Inference

Change the finetune field in. /configs/Res_train.yaml to False, and run script:

```
python test_ptbxl.py
# or
python test_georgia.py
# or
python test_ICBEB.py
# or
python test_shaoxing.py
# or
python clinical.py
```
## Fine-tune

Change the finetune field in ./configs/Res_train.yaml to True, and run script:

```
python test_ptbxl.py
# or
python test_georgia.py
# or
python test_ICBEB.py
# or
python test_shaoxing.py
# or
python clinical.py
```
Change the finetune_sample_rate field in ./configs/Res_train.yaml to 0.01, 0.1 or 1,  realize the fine-tuning of different conditions
