import json
import logging
import math
import os
import cv2
import time
import numpy as np
import random

from PIL import Image
from contextlib import suppress
from itertools import chain
from sklearn.metrics import roc_auc_score,accuracy_score,recall_score, matthews_corrcoef, f1_score
import csv
import pickle

import torch
import torch.nn.functional as F
from torch import nn

from factory import utils
from factory.loss import ClipLoss, UniCL

try:
    import wandb
except ImportError:
    wandb = None

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features

def train(model, ecg_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler, args, config, writer):
    clip_loss = ClipLoss(temperature=config["temperature"])
    uniCl = UniCL(temperature=config["temperature"], uniCl_type=config["uniCl_type"])

    ce_loss = nn.CrossEntropyLoss(ignore_index=-1)

    loss_m = AverageMeter()
    loss_clip_m = AverageMeter()
    loss_ce_m = AverageMeter()
    loss_ce_image_m = AverageMeter()
    loss_ce_text_m = AverageMeter()
    batch_time_m = AverageMeter()
    data_time_m = AverageMeter()
    end = time.time()

    model.train()  
    ecg_encoder.train()
    text_encoder.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_image', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_ce_text', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.add_meter('loss_clip', utils.SmoothedValue(window_size=50, fmt='{value:.6f}'))
    metric_logger.update(loss=1.0)
    metric_logger.update(lr = scheduler._get_lr(epoch)[0])

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50   
    step_size = 100
    warmup_iterations = warmup_steps*step_size 
    scalar_step = epoch*len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))
    if config["use_what_label"] == "diagnosis_label":
        if config["use_label_augment"]:
            with open("/home/tyy/unECG/dataset/all_label_augment.json", 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/unECG/dataset/all_diagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            text_list = []
            for key, value in all_diagnosis_label_map.items():
                item = background_info[value] + "This electrocardiogram diagnosed:" + value
                text_list.append(item)
        else:
            with open("/home/tyy/unECG/dataset/all_diagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            f = open('/home/tyy/ecg_ptbxl/output/exp1/data/mlb.pkl', 'rb')
            data = pickle.load(f)
            text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "subdiagnosis_label":
        if config["use_label_augment"]:
            with open("/home/tyy/unECG/dataset/label_augment_23.json", 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/unECG/dataset/all_subdiagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            text_list = []
            for key, value in all_diagnosis_label_map.items():
                item = background_info[value] + "This electrocardiogram diagnosed:" + value
                text_list.append(item)
        else:
            all_diagnosis_label_map = {'NORM': "normal ECG",
                                       'IMI': "inferior myocardial infarction",
                                       'AMI': "anterior myocardial infarction",
                                       'STTC': "ST/T-changes",
                                       'LVH': "left ventricular hypertrophy",
                                       'LAFB/LPFB': "left anterior/posterior fascicular block",
                                       'ISC_': "non-specific ischemic",
                                       'IRBBB': "incomplete right bundle branch block",
                                       'ISCA': "ischemic in anterolateral/anteroseptal/anterior leads",
                                       '_AVB': "AV block",
                                       'IVCD': "non-specific intraventricular conduction disturbance (block)",
                                       'NST_': "non-specific ST changes",
                                       'CRBBB': "complete right bundle branch block",
                                       'CLBBB': "complete left bundle branch block",
                                       'LAO/LAE': "left atrial overload/enlargement",
                                       'ISCI': "ischemic in inferior/inferolateral leads",
                                       'LMI': "lateral myocardial infarction",
                                       'RVH': "right ventricular hypertrophy",
                                       'RAO/RAE': "right atrial overload/enlargement",
                                       'WPW': "Wolf-Parkinson-White syndrome",
                                       'ILBBB': "incomplete left bundle branch block",
                                       'SEHYP': "septal hypertrophy",
                                       'PMI': "posterior myocardial infarction"}
            f = open('/home/tyy/ecg_ptbxl/output/exp1.1/data/mlb.pkl', 'rb')
            data = pickle.load(f)
            text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "form":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp2/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "rhythm":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp3/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "all":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp0/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_label_augment"]:
        backgroud_info = {
            "Myocardial Infarction": "To identify myocardial infarction on a 12-lead ECG, focus on leads II, III, and aVF to "
                                     "look for ST-segment elevation or depression. Additionally, look for reciprocal changes "
                                     "in leads V1-V4. ST-segment elevation in leads V1-V4 may indicate an anterior wall myocardial "
                                     "infarction, while ST-segment changes in leads II, III, and aVF may indicate an inferior "
                                     "wall myocardial infarction. Q waves may also be present in the affected leads.",
            "ST/T change": "To identify ST/T changes on a 12-lead ECG, the most important leads to focus on are leads II, "
                           "III, aVF, V5, and V6. Look for abnormalities such as ST-segment elevation or depression, T-wave "
                           "inversion or flattening, and QTc prolongation. Pay attention to the morphology and configuration "
                           "of the changes. Other leads may also be helpful, such as lead aVL for detecting lateral wall changes "
                           "and leads V1 and V2 for septal changes. It's important to also consider the patient's clinical "
                           "presentation and other factors when interpreting ECG findings.",
            "Conduction Disturbance": "In identifying conduction disturbances from a 12-lead ECG, you need to focus on the PR "
                                      "interval and the QRS duration.  A prolonged PR interval indicates first-degree AV block "
                                      "while a short PR interval suggests a possible Wolff-Parkinson-White (WPW) syndrome. "
                                      "A widened QRS can indicate bundle branch block, while a narrow QRS suggests normal conduction. "
                                      "Additionally, examining leads V1 and V6 can provide more context, as any deviations from their "
                                      "expected patterns can suggest specific conduction abnormalities.",
            "Hypertrophy": "To identify hypertrophy from a 12-lead ECG, you should focus on the QRS complex.  Specifically, "
                           "look for an increase in the amplitude of the QRS complex, which can suggest ventricular hypertrophy. "
                           "You should also examine leads V1 and V2, as a prominent R wave in these leads may indicate right "
                           "ventricular hypertrophy, while a prominent S wave in leads V5 and V6 may suggest left ventricular "
                           "hypertrophy.  Be sure to compare the amplitudes of the QRS complexes across all leads to make a "
                           "definitive diagnosis."}
        text_list = ["This electrocardiogram diagnosed: Normal ECG",
                     backgroud_info["Myocardial Infarction"] + "This electrocardiogram diagnosed: Myocardial Infarction",
                     backgroud_info["ST/T change"] + "This electrocardiogram diagnosed: ST/T change",
                     backgroud_info["Conduction Disturbance"] + "This electrocardiogram diagnosed: Conduction Disturbance",
                     backgroud_info["Hypertrophy"] + "This electrocardiogram diagnosed: Hypertrophy"]
    else:
        text_list = ["Normal ECG", "Myocardial Infarction", "ST/T change", "Conduction Disturbance",
                           "Hypertrophy"]
    try:
        for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            signal = sample['signal'].to(device)
            if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
                signal = signal.unsqueeze(1) #for lstm and resnet
            elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', 'xresnet1d_101']:
                signal = signal.transpose(1, 2) #for lstm and resnet
            elif config['ecg_model_name'] == 'swinT':
                signal = signal.transpose(1,2)

            label = sample['label'].to(device)
            label = torch.tensor(label, device=device)

            data_time_m.update(time.time() - end)
            optimizer.zero_grad()
            if config["ecg_model_name"] in ['resnet1d_wang', 'xresnet1d_101']:
                ecg_features = ecg_encoder(signal)
                ecg_features_pool = ecg_features.mean(-1)
            elif config["ecg_model_name"] in ['swinT']:
                ecg_features = ecg_encoder(signal)
            else:
                ecg_features, ecg_features_pool = ecg_encoder(signal)  # (32, 768, 300), (32, 768)

            if config["use_ecgNet_Diagnosis"] in ["ecgNet", "swinT"]:
                label = label.float()
                loss = ce_loss(ecg_features, label)
                loss_ce = loss
                loss_ce_ecg = loss
                loss_ce_text = loss
                loss_clip = loss
            elif config["use_ecgNet_Diagnosis"] == "engNet&TQN":
                """"""
                label_features = get_text_features(text_encoder, text_list, tokenizer, device,
                                                   max_length=args.max_length)  # (5,768)
                label = label.long()
                pred_class_ecg = model(ecg_features.transpose(1, 2), label_features)  # (32,40,2)
                loss = ce_loss(pred_class_ecg.view(-1, 2), label.view(-1))  # (16, 5, 2),(16, 5)
                loss_ce = loss
                loss_ce_ecg = loss
                loss_ce_text = loss
                loss_clip = loss
            else:
                label_features = get_text_features(text_encoder, text_list, tokenizer, device,
                                                   max_length=args.max_length)  # (5,768)
                report = sample['report']  # (32)
                report_features = get_text_features(text_encoder, report, tokenizer, device,
                                                    max_length=args.max_length)  # (32,768)

                label = label.long()
                pred_class_ecg = model(ecg_features.transpose(1, 2),label_features)  # (32,40,2)
                loss_ce_ecg = ce_loss(pred_class_ecg.view(-1,2),label.view(-1))     # (16, 5, 2),(16, 5)

                pred_class_text = model(report_features.unsqueeze(1), label_features)
                loss_ce_text = ce_loss(pred_class_text.view(-1, 2), label.view(-1))
                if config["loss_cross_image_text"]:
                    loss_ce = loss_ce_ecg if random.random() > 0.5 else loss_ce_text
                else:
                    loss_ce = loss_ce_ecg * config["loss_ratio"] + loss_ce_text

                if config["loss_type"] == "uniCl":
                    loss_clip = uniCl(ecg_features_pool,report_features, label)
                else:
                    loss_clip = clip_loss(ecg_features_pool,report_features) # (32, 768) (32,768)
                loss = loss_ce * config["loss_ratio"] + loss_clip
            loss.backward()
            optimizer.step()

            writer.add_scalar('loss/loss', loss, scalar_step)
            writer.add_scalar('loss/loss_ce', loss_ce, scalar_step)
            writer.add_scalar('loss/loss_ce_ecg', loss_ce_ecg, scalar_step)
            writer.add_scalar('loss/loss_ce_text', loss_ce_text, scalar_step)
            writer.add_scalar('loss/loss_clip', loss_clip, scalar_step)
            scalar_step += 1

            metric_logger.update(loss=loss.item())
            metric_logger.update(loss_ce=loss_ce.item())
            metric_logger.update(loss_ce_image=loss_ce_ecg.item())
            metric_logger.update(loss_ce_text=loss_ce_text.item())
            metric_logger.update(loss_clip=loss_clip.item())


            if epoch==0 and i%step_size==0 and i<=warmup_iterations:
                scheduler.step(i//step_size)
            metric_logger.update(lr = scheduler._get_lr(epoch)[0])

            batch_time_m.update(time.time() - end)
            end = time.time()
            batch_count = i + 1
            if i % 100 == 0:
                batch_size = len(signal)
                num_samples = batch_count * batch_size
                samples_per_epoch = data_loader.num_samples
                percent_complete = 100.0 * batch_count / num_batches_per_epoch

                # NOTE loss is coarsely sampled, just master node and per log update
                loss_m.update(loss.item(), batch_size)
                loss_clip_m.update(loss_clip.item(), batch_size)
                loss_ce_m.update(loss_ce.item(), batch_size)
                loss_ce_image_m.update(loss_ce_ecg.item(), batch_size)
                loss_ce_text_m.update(loss_ce_text.item(), batch_size)

                logging.info(
                    f"Train Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
                    f"Loss: {loss_m.val:#.5g} ({loss_m.avg:#.4g}) "
                    f"Loss_clip: {loss_clip_m.val:#.5g} ({loss_clip_m.avg:#.4g}) "
                    f"Loss_ce: {loss_ce_m.val:#.5g} ({loss_ce_m.avg:#.4g}) "
                    f"Loss_ce_image: {loss_ce_image_m.val:#.5g} ({loss_ce_image_m.avg:#.4g}) "
                    f"Loss_ce_text: {loss_ce_text_m.val:#.5g} ({loss_ce_text_m.avg:#.4g}) "
                    f"Data (t): {data_time_m.avg:.3f} "
                    f"Batch (t): {batch_time_m.avg:.3f}, {batch_size / batch_time_m.val:#g}/s "
                    f"LR: {scheduler._get_lr(epoch)[0]:5f} "
                )
    except Exception as e:
        print(e)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger.global_avg())     
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()} #,loss_epoch.mean()

def valid_on_ptb(model, ecg_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer):
    if config["use_ecgNet_Diagnosis"] == "ecgNet":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    ecg_encoder.eval()
    text_encoder.eval()
    if config["use_what_label"] == "diagnosis_label":
        if config["use_label_augment"]:
            with open("/home/tyy/unECG/dataset/all_label_augment.json", 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/unECG/dataset/all_diagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            text_list = []
            for key, value in all_diagnosis_label_map.items():
                item = background_info[value] + "This electrocardiogram diagnosed:" + value
                text_list.append(item)
        else:
            with open("/home/tyy/unECG/dataset/all_diagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            f = open('/home/tyy/ecg_ptbxl/output/exp1/data/mlb.pkl', 'rb')
            data = pickle.load(f)
            text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "subdiagnosis_label":
        if config["use_label_augment"]:
            with open("/home/tyy/unECG/dataset/all_subdiagnosis_label_augment.json", 'r') as f:
                background_info = json.load(f)
            with open("/home/tyy/unECG/dataset/all_subdiagnosis_label_map.json", 'r') as f:
                all_diagnosis_label_map = json.load(f)
            text_list = []
            for key, value in all_diagnosis_label_map.items():
                item = background_info[value] + "This electrocardiogram diagnosed:" + value
                text_list.append(item)
        else:
            all_diagnosis_label_map = {'NORM':"normal ECG",
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
            f = open('/home/tyy/ecg_ptbxl/output/exp1.1/data/mlb.pkl', 'rb')
            data = pickle.load(f)
            text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "all":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp0/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "form":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp2/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_what_label"] == "rhythm":
        with open("/home/tyy/unECG/dataset/all_label_map.json", 'r') as f:
            all_diagnosis_label_map = json.load(f)
        f = open('/home/tyy/ecg_ptbxl/output/exp3/data/mlb.pkl', 'rb')
        data = pickle.load(f)
        text_list = [all_diagnosis_label_map[item] for item in data.classes_]
    elif config["use_label_augment"]:
        backgroud_info = {
            "Myocardial Infarction": "To identify myocardial infarction on a 12-lead ECG, focus on leads II, III, and aVF to "
                                     "look for ST-segment elevation or depression. Additionally, look for reciprocal changes "
                                     "in leads V1-V4. ST-segment elevation in leads V1-V4 may indicate an anterior wall myocardial "
                                     "infarction, while ST-segment changes in leads II, III, and aVF may indicate an inferior "
                                     "wall myocardial infarction. Q waves may also be present in the affected leads.",
            "ST/T change": "To identify ST/T changes on a 12-lead ECG, the most important leads to focus on are leads II, "
                           "III, aVF, V5, and V6. Look for abnormalities such as ST-segment elevation or depression, T-wave "
                           "inversion or flattening, and QTc prolongation. Pay attention to the morphology and configuration "
                           "of the changes. Other leads may also be helpful, such as lead aVL for detecting lateral wall changes "
                           "and leads V1 and V2 for septal changes. It's important to also consider the patient's clinical "
                           "presentation and other factors when interpreting ECG findings.",
            "Conduction Disturbance": "In identifying conduction disturbances from a 12-lead ECG, you need to focus on the PR "
                                      "interval and the QRS duration.  A prolonged PR interval indicates first-degree AV block "
                                      "while a short PR interval suggests a possible Wolff-Parkinson-White (WPW) syndrome. "
                                      "A widened QRS can indicate bundle branch block, while a narrow QRS suggests normal conduction. "
                                      "Additionally, examining leads V1 and V6 can provide more context, as any deviations from their "
                                      "expected patterns can suggest specific conduction abnormalities.",
            "Hypertrophy": "To identify hypertrophy from a 12-lead ECG, you should focus on the QRS complex.  Specifically, "
                           "look for an increase in the amplitude of the QRS complex, which can suggest ventricular hypertrophy. "
                           "You should also examine leads V1 and V2, as a prominent R wave in these leads may indicate right "
                           "ventricular hypertrophy, while a prominent S wave in leads V5 and V6 may suggest left ventricular "
                           "hypertrophy.  Be sure to compare the amplitudes of the QRS complexes across all leads to make a "
                           "definitive diagnosis."}
        text_list = ["This electrocardiogram diagnosed: Normal ECG",
                     backgroud_info[
                         "Myocardial Infarction"] + "This electrocardiogram diagnosed: Myocardial Infarction",
                     backgroud_info["ST/T change"] + "This electrocardiogram diagnosed: ST/T change",
                     backgroud_info[
                         "Conduction Disturbance"] + "This electrocardiogram diagnosed: Conduction Disturbance",
                     backgroud_info["Hypertrophy"] + "This electrocardiogram diagnosed: Hypertrophy"]
    else:
        text_list = ["Normal ECG", "Myocardial Infarction", "ST/T change", "Conduction Disturbance",
                     "Hypertrophy"]

    
    val_scalar_step = epoch*len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda(device=device)
    pred = torch.FloatTensor()
    pred = pred.cuda(device=device)

    for i, sample in enumerate(data_loader):
        signal = sample['signal'].to(device)
        if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
            signal = signal.unsqueeze(1) #for lstm and resnet
        elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', 'xresnet1d_101']:
            signal = signal.transpose(1, 2) #for lstm and resnet
        elif config['ecg_model_name'] == 'swinT':
            signal = signal.transpose(1, 2)  # for lstm and resnet

        label = sample['label'].to(device)
        label = torch.tensor(label, device=device)

        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            if config["ecg_model_name"] in ['resnet1d_wang', 'xresnet1d_101']:
                ecg_features = ecg_encoder(signal)
                ecg_features_pool = ecg_features.mean(-1)
            elif config["ecg_model_name"] in ['swinT']:
                ecg_features = ecg_encoder(signal)
            else:
                ecg_features, ecg_features_pool = ecg_encoder(signal)  # (32, 768, 300), (32, 768)

            if config["use_ecgNet_Diagnosis"] in ["ecgNet", "swinT"]:
                """"""
                pred = torch.cat((pred, ecg_features), 0)
                label = label.float()
                val_loss = criterion(ecg_features, label)
            else:
                label = label.long()
                text_features = get_text_features(text_encoder, text_list, tokenizer, device,
                                                  max_length=args.max_length)
                pred_class = model(ecg_features.transpose(1, 2), text_features)  # (64,5,2)
                val_loss = criterion(pred_class.view(-1,2),label.view(-1))
                pred_class = torch.softmax(pred_class, dim=-1)
                pred = torch.cat((pred, pred_class[:,:,1]), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt, pred, n_class = config["class_num"])
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()

    mccs, threshold = compute_mccs(gt.cpu().numpy(), pred.cpu().numpy(), n_class=config["class_num"])
    F1s = compute_F1s_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=config["class_num"])
    Accs = compute_Accs_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=config["class_num"])
    metrics["F1"] = F1s[-1]
    metrics["Accs"] = Accs[-1]
    return avg_val_loss,AUROC_avg,metrics

def compute_F1s_threshold(gt, pred, threshold, n_class=6):
    bert_f1 = 0.0
    gt_np = gt
    pred_np = pred

    F1s = []
    F1s.append('F1s')
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        F1s.append(f1_score(gt_np[:, i], pred_np[:, i], average='binary'))
    mean_f1 = np.mean(np.array(F1s[1:]))
    F1s.append(mean_f1)
    return F1s


def compute_Accs_threshold(gt, pred, threshold, n_class=6):
    gt_np = gt
    pred_np = pred

    Accs = []
    Accs.append('Accs')
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
        Accs.append(accuracy_score(gt_np[:, i], pred_np[:, i]))
    mean_accs = np.mean(np.array(Accs[1:]))
    Accs.append(mean_accs)
    return Accs


def compute_mccs(gt, pred, n_class=6):
    # get a best threshold for all classes
    gt_np = gt
    pred_np = pred
    select_best_thresholds = []
    best_mcc = 0.0

    for i in range(n_class):
        select_best_threshold_i = 0.0
        best_mcc_i = 0.0
        for threshold_idx in range(len(pred)):
            pred_np_ = pred_np.copy()
            thresholds = pred[threshold_idx]
            pred_np_[:, i][pred_np_[:, i] >= thresholds[i]] = 1
            pred_np_[:, i][pred_np_[:, i] < thresholds[i]] = 0
            mcc = matthews_corrcoef(gt_np[:, i], pred_np_[:, i])
            if mcc > best_mcc_i:
                best_mcc_i = mcc
                select_best_threshold_i = thresholds[i]
        select_best_thresholds.append(select_best_threshold_i)
    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= select_best_thresholds[i]] = 1
        pred_np[:, i][pred_np[:, i] < select_best_thresholds[i]] = 0
    mccs = []
    mccs.append('mccs')
    for i in range(n_class):
        mccs.append(matthews_corrcoef(gt_np[:, i], pred_np[:, i]))
    mean_mcc = np.mean(np.array(mccs[1:]))
    mccs.append(mean_mcc)
    return mccs, select_best_thresholds

import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    metrics = {}
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    if n_class == 5:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
    if n_class == 23:
        metrics[f"auc/class_0"] = AUROCs[0]
        # metrics[f"auc/class_1"]=AUROCs[1]
        # metrics[f"auc/class_2"]=AUROCs[2]
        # metrics[f"auc/class_3"]=AUROCs[3]
        # metrics[f"auc/class_4"]=AUROCs[4]
        # metrics[f"auc/class_5"] = AUROCs[5]
        metrics[f"auc/class_6"]=AUROCs[6]
        # metrics[f"auc/class_7"]=AUROCs[7]
        # metrics[f"auc/class_8"]=AUROCs[8]
        # metrics[f"auc/class_9"]=AUROCs[9]
        # metrics[f"auc/class_10"]=AUROCs[10]
        # metrics[f"auc/class_11"]=AUROCs[11]
        metrics[f"auc/class_12"]=AUROCs[12]
        # metrics[f"auc/class_13"]=AUROCs[13]
        # metrics[f"auc/class_14"] = AUROCs[14]
        # metrics[f"auc/class_15"] = AUROCs[15]
        # metrics[f"auc/class_16"] = AUROCs[16]
        # metrics[f"auc/class_17"] = AUROCs[17]
        metrics[f"auc/class_18"] = AUROCs[18]
        # metrics[f"auc/class_19"] = AUROCs[19]
        # metrics[f"auc/class_20"] = AUROCs[20]
        # metrics[f"auc/class_21"] = AUROCs[21]
        metrics[f"auc/class_22"] = AUROCs[22]
    elif n_class == 44:
        metrics[f"auc/class_0"]=AUROCs[0]
        # metrics[f"auc/class_1"]=AUROCs[1]
        # metrics[f"auc/class_2"]=AUROCs[2]
        # metrics[f"auc/class_3"]=AUROCs[3]
        # metrics[f"auc/class_4"]=AUROCs[4]
        metrics[f"auc/class_5"]=AUROCs[5]
        # metrics[f"auc/class_6"]=AUROCs[6]
        # metrics[f"auc/class_7"]=AUROCs[7]
        # metrics[f"auc/class_8"]=AUROCs[8]
        # metrics[f"auc/class_9"]=AUROCs[9]
        # metrics[f"auc/class_10"]=AUROCs[10]
        # metrics[f"auc/class_11"]=AUROCs[11]
        # metrics[f"auc/class_12"]=AUROCs[12]
        # metrics[f"auc/class_13"]=AUROCs[13]
        # metrics[f"auc/class_14"] = AUROCs[14]
        # metrics[f"auc/class_15"] = AUROCs[15]
        # metrics[f"auc/class_16"] = AUROCs[16]
        # metrics[f"auc/class_17"] = AUROCs[17]
        metrics[f"auc/class_18"] = AUROCs[18]
        metrics[f"auc/class_19"] = AUROCs[19]
        # metrics[f"auc/class_20"] = AUROCs[20]
        # metrics[f"auc/class_21"] = AUROCs[21]
        # metrics[f"auc/class_22"] = AUROCs[22]
        # metrics[f"auc/class_23"] = AUROCs[23]
        # metrics[f"auc/class_24"] = AUROCs[24]
        # metrics[f"auc/class_25"] = AUROCs[25]
        # metrics[f"auc/class_26"] = AUROCs[26]
        # metrics[f"auc/class_27"] = AUROCs[27]
        # metrics[f"auc/class_28"] = AUROCs[28]
        metrics[f"auc/class_29"] = AUROCs[29]
        # metrics[f"auc/class_30"] = AUROCs[20]
        # metrics[f"auc/class_31"] = AUROCs[31]
        # metrics[f"auc/class_32"] = AUROCs[32]
        # metrics[f"auc/class_33"] = AUROCs[33]
        # metrics[f"auc/class_34"] = AUROCs[34]
        # metrics[f"auc/class_35"] = AUROCs[35]
        # metrics[f"auc/class_36"] = AUROCs[36]
        # metrics[f"auc/class_37"] = AUROCs[37]
        # metrics[f"auc/class_38"] = AUROCs[38]
        # metrics[f"auc/class_39"] = AUROCs[39]
        # metrics[f"auc/class_40"] = AUROCs[40]
        # metrics[f"auc/class_41"] = AUROCs[41]
        # metrics[f"auc/class_42"] = AUROCs[42]
        metrics[f"auc/class_43"] = AUROCs[43]
    elif n_class == 71:
        metrics[f"auc/class_0"]=AUROCs[0]
        metrics[f"auc/class_1"]=AUROCs[1]
        metrics[f"auc/class_2"]=AUROCs[2]
        metrics[f"auc/class_3"]=AUROCs[3]
        metrics[f"auc/class_4"]=AUROCs[4]
        metrics[f"auc/class_5"]=AUROCs[5]
        metrics[f"auc/class_6"]=AUROCs[6]
        metrics[f"auc/class_7"]=AUROCs[7]
        metrics[f"auc/class_8"]=AUROCs[8]
        metrics[f"auc/class_9"]=AUROCs[9]
        metrics[f"auc/class_10"]=AUROCs[10]
        metrics[f"auc/class_11"]=AUROCs[11]
        metrics[f"auc/class_12"]=AUROCs[12]
        metrics[f"auc/class_13"]=AUROCs[13]
        metrics[f"auc/class_14"] = AUROCs[14]
        metrics[f"auc/class_15"] = AUROCs[15]
        metrics[f"auc/class_16"] = AUROCs[16]
        metrics[f"auc/class_17"] = AUROCs[17]
        metrics[f"auc/class_18"] = AUROCs[18]
        metrics[f"auc/class_19"] = AUROCs[19]
        metrics[f"auc/class_20"] = AUROCs[20]
        metrics[f"auc/class_21"] = AUROCs[21]
        metrics[f"auc/class_22"] = AUROCs[22]
        metrics[f"auc/class_23"] = AUROCs[23]
        metrics[f"auc/class_24"] = AUROCs[24]
        metrics[f"auc/class_25"] = AUROCs[25]
        metrics[f"auc/class_26"] = AUROCs[26]
        metrics[f"auc/class_27"] = AUROCs[27]
        metrics[f"auc/class_28"] = AUROCs[28]
        metrics[f"auc/class_29"] = AUROCs[29]
        metrics[f"auc/class_30"] = AUROCs[20]
        metrics[f"auc/class_31"] = AUROCs[31]
        metrics[f"auc/class_32"] = AUROCs[32]
        metrics[f"auc/class_33"] = AUROCs[33]
        metrics[f"auc/class_34"] = AUROCs[34]
        metrics[f"auc/class_35"] = AUROCs[35]
        metrics[f"auc/class_36"] = AUROCs[36]
        metrics[f"auc/class_37"] = AUROCs[37]
        metrics[f"auc/class_38"] = AUROCs[38]
        metrics[f"auc/class_39"] = AUROCs[39]
        metrics[f"auc/class_40"] = AUROCs[40]
        metrics[f"auc/class_41"] = AUROCs[41]
        metrics[f"auc/class_42"] = AUROCs[42]
        metrics[f"auc/class_43"] = AUROCs[43]
        metrics[f"auc/class_44"] = AUROCs[44]
        metrics[f"auc/class_45"] = AUROCs[45]
        metrics[f"auc/class_46"] = AUROCs[46]
        metrics[f"auc/class_47"] = AUROCs[47]
        metrics[f"auc/class_48"] = AUROCs[48]
        metrics[f"auc/class_49"] = AUROCs[49]
        metrics[f"auc/class_50"] = AUROCs[50]
        metrics[f"auc/class_51"] = AUROCs[51]
        metrics[f"auc/class_52"] = AUROCs[52]
        metrics[f"auc/class_53"] = AUROCs[53]
        metrics[f"auc/class_54"] = AUROCs[54]
        metrics[f"auc/class_55"] = AUROCs[55]
        metrics[f"auc/class_56"] = AUROCs[56]
        metrics[f"auc/class_57"] = AUROCs[57]
        metrics[f"auc/class_58"] = AUROCs[58]
        metrics[f"auc/class_59"] = AUROCs[59]
        metrics[f"auc/class_60"] = AUROCs[60]
        metrics[f"auc/class_61"] = AUROCs[61]
        metrics[f"auc/class_62"] = AUROCs[62]
        metrics[f"auc/class_63"] = AUROCs[63]
        metrics[f"auc/class_64"] = AUROCs[64]
        metrics[f"auc/class_65"] = AUROCs[65]
        metrics[f"auc/class_66"] = AUROCs[66]
        metrics[f"auc/class_67"] = AUROCs[67]
        metrics[f"auc/class_68"] = AUROCs[68]
        metrics[f"auc/class_69"] = AUROCs[69]
        metrics[f"auc/class_70"] = AUROCs[70]
    return metrics




