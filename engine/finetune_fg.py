# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-07-11 22:23

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
from sklearn.metrics import roc_auc_score, accuracy_score, recall_score, matthews_corrcoef, f1_score
from factory.visualization import visualization_tsne
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


def get_text_features(model, text_list, tokenizer, device, max_length):
    text_token = tokenizer(list(text_list), add_special_tokens=True, max_length=max_length, pad_to_max_length=True,
                           return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features


def finetune(model, ecg_encoder, text_encoder, tokenizer, data_loader, optimizer, epoch, warmup_steps, device, scheduler,
          args, config, writer, text_list):
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
    metric_logger.update(lr=scheduler._get_lr(epoch)[0])

    header = 'Finetune Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size
    scalar_step = epoch * len(data_loader)
    num_batches_per_epoch = data_loader.num_batches
    sample_digits = math.ceil(math.log(data_loader.num_samples + 1, 10))
    try:
        for i, sample in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
            # signal = sample['signal'].cuda(non_blocking=True)
            signal = sample['signal'].to(device)
            if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
                signal = signal.unsqueeze(1)  # for lstm and resnet
            elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', 'xresnet1d_101']:
                signal = signal.transpose(1, 2)  # for lstm and resnet

            label = sample['label'].to(device)
            label = torch.tensor(label, device=device)

            # label_name = sample['label_name'] # (32)
            data_time_m.update(time.time() - end)
            optimizer.zero_grad()
            if config["ecg_model_name"] in ['resnet1d_wang', 'xresnet1d_101']:
                ecg_features = ecg_encoder(signal)
                ecg_features_pool = ecg_features.mean(-1)
            else:
                ecg_features, ecg_features_pool = ecg_encoder(signal)  # (32, 768, 300), (32, 768)

            label_features = get_text_features(text_encoder, text_list, tokenizer, device,
                                               max_length=args.max_length)  # (5,768)
            report = sample['report']  # (32)
            report_features = get_text_features(text_encoder, report, tokenizer, device,
                                                max_length=args.max_length)  # (32,768)

            label = label.long()
            pred_class_ecg = model(ecg_features.transpose(1, 2), label_features)  # (32,40,2)
            loss_ce_ecg = ce_loss(pred_class_ecg.view(-1, 2), label.view(-1))  # (16, 5, 2),(16, 5)

            pred_class_text = model(report_features.unsqueeze(1), label_features)
            loss_ce_text = ce_loss(pred_class_text.view(-1, 2), label.view(-1))
            if config["loss_cross_image_text"]:
                loss_ce = loss_ce_ecg if random.random() > 0.5 else loss_ce_text
            else:
                loss_ce = loss_ce_ecg * config["loss_ratio"] + loss_ce_text

            if config["loss_type"] == "uniCl":
                loss_clip = uniCl(ecg_features_pool, report_features, label)
            else:
                loss_clip = clip_loss(ecg_features_pool, report_features)  # (32, 768) (32,768)
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

            if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
                scheduler.step(i // step_size)
            metric_logger.update(lr=scheduler._get_lr(epoch)[0])

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
                    f"Finetune Epoch: {epoch} [{num_samples:>{sample_digits}}/{samples_per_epoch} ({percent_complete:.0f}%)] "
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
    return {k: "{:.6f}".format(meter.global_avg) for k, meter in metric_logger.meters.items()}  # ,loss_epoch.mean()


def valid_finetune(model, ecg_encoder, text_encoder, tokenizer, data_loader, epoch, device, args, config, writer, text_list):
    if config["use_ecgNet_Diagnosis"] == "ecgNet":
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    model.eval()
    ecg_encoder.eval()
    text_encoder.eval()
    text_features = get_text_features(text_encoder, text_list, tokenizer, device, max_length=args.max_length)

    val_scalar_step = epoch * len(data_loader)
    val_losses = []

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    gt = gt.cuda(device=device)
    pred = torch.FloatTensor()
    pred = pred.cuda(device=device)

    for i, sample in enumerate(data_loader):
        signal = sample['signal'].to(device)
        if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
            signal = signal.unsqueeze(1)  # for lstm and resnet
        elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', 'xresnet1d_101']:
            signal = signal.transpose(1, 2)  # for lstm and resnet
        label = sample['label'].to(device)
        label = torch.tensor(label, device=device)

        gt = torch.cat((gt, label), 0)
        with torch.no_grad():
            if config["ecg_model_name"] in ['resnet1d_wang', 'xresnet1d_101']:
                ecg_features = ecg_encoder(signal)
                ecg_features_pool = ecg_features.mean(-1)
            else:
                ecg_features, ecg_features_pool = ecg_encoder(signal)  # (32, 768, 300), (32, 768)
            if config['visualization'] == 'resnet':
                visualization_tsne(ecg_features_pool, label)
            elif config['visualization'] == 'lqn':
                model.visual_tsne(ecg_features.transpose(1, 2), text_features,label)
            if config["use_ecgNet_Diagnosis"] == "ecgNet":
                """"""
                pred = torch.cat((pred, ecg_features), 0)
                label = label.float()
                val_loss = criterion(ecg_features, label)
            else:
                label = label.long()
                pred_class = model(ecg_features.transpose(1, 2), text_features)  # (64,5,2)
                val_loss = criterion(pred_class.view(-1, 2), label.view(-1))
                pred_class = torch.softmax(pred_class, dim=-1)
                pred = torch.cat((pred, pred_class[:, :, 1]), 0)
            val_losses.append(val_loss.item())
            writer.add_scalar('val_loss/loss', val_loss, val_scalar_step)
            val_scalar_step += 1
    metrics = compute_AUCs(gt.cpu().numpy(), pred.cpu().numpy(), n_class=len(text_list))
    AUROC_avg = metrics['mean_auc']
    avg_val_loss = np.array(val_losses).mean()

    mccs, threshold = compute_mccs(gt.cpu().numpy(), pred.cpu().numpy(), n_class=len(text_list))
    F1s = compute_F1s_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=len(text_list))
    Accs = compute_Accs_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=len(text_list))


    for idx in range(len(text_list)):
        metrics[f"mccs/class_{idx}"] = mccs[idx + 1]
    metrics["mccs"] = mccs[-1]
    for idx in range(len(text_list)):
        metrics[f"F1/class_{idx}"] = F1s[idx + 1]
    metrics["F1"] = F1s[-1]
    for idx in range(len(text_list)):
        metrics[f"Accs/class_{idx}"] = Accs[idx + 1]
    metrics["Accs"] = Accs[-1]

    confidence_result = []
    if config['getConfidence']:
        confidence_result = get_confidence(gt, pred, num_class=len(text_list))

    return avg_val_loss, AUROC_avg, metrics, confidence_result

def get_confidence(gt, pred, num_class):
    AUROCs_list = []
    mccs_list = []
    F1s_list = []
    Accs_list = []
    output_list = []

    data_len = len(gt)
    for idx in range(1000):
        randnum = random.randint(0, 5000)
        random.seed(randnum)
        gt_idx = random.choices(gt.cpu().numpy(), k=data_len)
        random.seed(randnum)
        pred_idx = random.choices(pred.cpu().numpy(), k=data_len)
        gt_idx = np.array(gt_idx)
        pred_idx = np.array(pred_idx)

        AUROCs_idx = compute_AUCs_confidence(gt_idx, pred_idx, n_class=num_class)
        mccs_idx, threshold_idx = compute_mccs(gt_idx, pred_idx, n_class=num_class)
        F1s_idx = compute_F1s_threshold(gt_idx, pred_idx, threshold_idx, n_class=num_class)
        Accs_idx = compute_Accs_threshold(gt_idx, pred_idx, threshold_idx, n_class=num_class)

        AUROCs_list.append(AUROCs_idx[1:])  # 1000,5
        mccs_list.append(mccs_idx[1:])
        F1s_list.append(F1s_idx[1:])
        Accs_list.append(Accs_idx[1:])
    AUROCs_5, AUROCs_95, AUROCs_mean = get_sort_eachclass(AUROCs_list, n_class=num_class)
    output = []
    output.append('perclass_AUROCs_5')
    output.extend(AUROCs_5)
    output_list.append(output)
    output = []
    output.append('perclass_AUROCs_95')
    output.extend(AUROCs_95)
    output_list.append(output)
    output = []
    output.append('perclass_AUROCs_mean')
    output.extend(AUROCs_mean)
    output_list.append(output)

    mccs_5, mccs_95, mccs_mean = get_sort_eachclass(mccs_list, n_class=num_class)
    output = []
    output.append('perclass_mccs_5')
    output.extend(mccs_5)
    output_list.append(output)
    output = []
    output.append('perclass_mccs_95')
    output.extend(mccs_95)
    output_list.append(output)
    output = []
    output.append('perclass_mccs_mean')
    output.extend(mccs_mean)
    output_list.append(output)

    F1s_5, F1s_95, F1s_mean = get_sort_eachclass(F1s_list, n_class=num_class)
    output = []
    output.append('perclass_F1s_5')
    output.extend(F1s_5)
    output_list.append(output)
    output = []
    output.append('perclass_F1s_95')
    output.extend(F1s_95)
    output_list.append(output)
    output = []
    output.append('perclass_F1s_mean')
    output.extend(F1s_mean)
    output_list.append(output)

    Accs_5, Accs_95, Accs_mean = get_sort_eachclass(Accs_list, n_class=num_class)
    output = []
    output.append('perclass_Accs_5')
    output.extend(Accs_5)
    output_list.append(output)
    output = []
    output.append('perclass_Accs_95')
    output.extend(Accs_95)
    output_list.append(output)
    output = []
    output.append('perclass_Accs_mean')
    output.extend(Accs_mean)
    output_list.append(output)
    return output_list

def get_sort_eachclass(metric_list,n_class=6):
    metric_5=[]
    metric_95=[]
    metric_mean=[]
    for i in range(n_class):
        sorted_metric_list = sorted(metric_list,key=lambda x:x[i])
        metric_5.append(sorted_metric_list[50][i])
        metric_95.append(sorted_metric_list[950][i])
        metric_mean.append(np.mean(np.array(sorted_metric_list),axis=0)[i])
    mean_metric_5 = np.mean(np.array(metric_5))
    metric_5.append(mean_metric_5)
    mean_metric_95 = np.mean(np.array(metric_95))
    metric_95.append(mean_metric_95)
    mean_metric_mean = np.mean(np.array(metric_mean))
    metric_mean.append(mean_metric_mean)
    return metric_5,metric_95,metric_mean

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


def draw_confusion_matrix(ground_truth, pred_label, labels, epoch):
    sns.set()
    f, ax = plt.subplots()
    C2 = confusion_matrix(ground_truth.cpu().detach().numpy(), pred_label.cpu().detach().numpy(), labels=labels)
    # 打印 C2
    print(C2)
    sns.heatmap(C2, annot=True, ax=ax)  # 画热力图
    ax.set_title('valid_' + str(epoch))  # 标题
    ax.set_xlabel('predict')  # x 轴
    ax.set_ylabel('true')  # y 轴
    plt.show()

    # 将对角线上的元素设置为0
    np.fill_diagonal(C2, 0)
    # 找出除对角线之外的最大值
    max_value = np.max(C2)
    # 找出最大值的坐标
    indices = np.argwhere(C2 == max_value)
    # 输出结果
    return indices  # [[truth_label_index, pred_label_index][truth_label_index, pred_label_index]...]

def compute_AUCs_confidence(gt, pred, n_class=6):
    metrics = {}
    AUROCs = []
    AUROCs.append('AUC')
    gt_np = gt
    pred_np = pred
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    mean_auc = np.mean(np.array(AUROCs[1:]))
    AUROCs.append(mean_auc)
    return AUROCs

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
    gt_np = gt
    pred_np = pred
    for i in range(n_class):
        AUROCs.append(roc_auc_score(gt_np[:, i], pred_np[:, i]))
    metrics[f"mean_auc"] = np.mean(np.array(AUROCs))
    for idx in range(n_class):
        metrics[f"auc/class_{idx}"] = AUROCs[idx]
    return metrics


if __name__ == '__main__':
    draw_confusion_matrix([0, 0, 1, 2, 1, 2, 0, 2, 2, 0, 1, 1], [1, 0, 1, 2, 1, 0, 0, 2, 2, 0, 1, 1], [0, 1, 2], 10)
