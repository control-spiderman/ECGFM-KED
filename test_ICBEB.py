# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-10 18:30
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from tensorboardX import SummaryWriter

from factory import utils
from dataset.ecgDataset import FinetuneDataset
from models.ECGNet import ECGNet
from models.clip_model import CLP_clinical, TQNModel
from models.resnet1d_wang import resnet1d_wang
from models.xresnet1d_101 import xresnet1d101
from engine.finetune_fg import finetune, valid_finetune

from optim import create_optimizer
from scheduler import create_scheduler

import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import csv
from pathlib import Path
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score

def down_sample_train_data(X_train, y_train, sample_rate=1.0):
    num_samples = int(len(X_train) * sample_rate)
    indices = np.arange(len(X_train))
    random_indices = np.random.choice(indices, size=num_samples, replace=False)
    X_train_new = X_train[random_indices]
    y_train_new = y_train[random_indices]
    return X_train_new, y_train_new

def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    # seed = args.seed + utils.get_rank()
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    X = np.load("./dataset/cpsc/test_ICBEB_signal.npy", allow_pickle=True)
    if config['test_ICBEB_class_nums'] == 9:
        y = np.load("./dataset/cpsc/test_ICBEB_label_9.npy", allow_pickle=True)
    elif config['test_ICBEB_class_nums'] == 6:
        y = np.load("./dataset/cpsc/test_ICBEB_label_6.npy", allow_pickle=True)
    elif config['test_ICBEB_class_nums'] == 3:
        y = np.load("./dataset/cpsc/test_ICBEB_label.npy", allow_pickle=True)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=2, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=2, shuffle=True)
    print("test set number：", len(X_test))
    X_train, y_train = down_sample_train_data(X_train, y_train, config['finetune_sample_rate'])
    train_dataset = FinetuneDataset(X_train, y_train, "cpsc")
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['finetune_batch_size'],
                                  num_workers=0,
                                  sampler=None,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=None)
    test_dataset = FinetuneDataset(X_test, y_test, "cpsc")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['finetune_batch_size'],
                                 num_workers=0,
                                 sampler=None,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=None)
    val_dataset = FinetuneDataset(X_val, y_val, "cpsc")
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['finetune_batch_size'],
                                num_workers=0,
                                sampler=None,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=None)
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader)
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)
    if config["ecg_model_name"] == 'ecgNet':
        ecg_model = ECGNet(input_channel=1, use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'resnet1d_wang':
        ecg_model = resnet1d_wang(num_classes=config['test_ICBEB_class_nums'], input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[128], inplanes=768,use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'xresnet1d_101':
        ecg_model = xresnet1d101(num_classes=config["class_num"], input_channels=12, kernel_size=5,
                          ps_head=0.5).to(device=device)
    # load model
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name, freeze_layers=config['freeze_layers']).to(device=device)
    model = TQNModel(num_layers=config["tqn_model_layers"]).to(device)

    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    ecg_model_state_dict = checkpoint['ecg_model']
    ecg_model.load_state_dict(ecg_model_state_dict)

    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    text_encoder_state_dict = checkpoint["text_encoder"]
    text_encoder.load_state_dict(text_encoder_state_dict)
    print('load checkpoint from %s' % args.checkpoint)

    if config['finetune']:
        print("Start finetune")
        start_time = time.time()
        start_epoch = 0
        max_epoch = config['schedular']['finetune_epochs']
        warmup_steps = config['schedular']['warmup_epochs']
        writer = SummaryWriter(os.path.join(args.finetune_output_dir, 'log'))
        arg_opt = utils.AttrDict(config['optimizer'])
        optimizer = create_optimizer(arg_opt, model, ecg_model, text_encoder)
        arg_sche = utils.AttrDict(config['schedular'])
        lr_scheduler, _ = create_scheduler(arg_sche, optimizer)
        best_val_auc = 0.0
        with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
            f.write(config["finetune_purpose"] + "\n")
        for epoch in range(start_epoch, max_epoch):
            if epoch > 0:
                lr_scheduler.step(epoch + warmup_steps)
            finetune_stats = finetune(model, ecg_model, text_encoder, tokenizer, train_dataloader, optimizer, epoch,
                                      warmup_steps, device, lr_scheduler, args, config, writer,
                                      train_dataset.label_name)
            for k, v in finetune_stats.items():
                if k == 'loss':
                    finetune_loss_epoch = v
                elif k == 'loss_ce':
                    finetune_loss_ce_epoch = v
                elif k == 'loss_clip':
                    finetune_loss_clip_epoch = v

            writer.add_scalar('loss/finetune_loss_epoch', float(finetune_loss_epoch), epoch)
            writer.add_scalar('loss/finetune_loss_ce_epoch', float(finetune_loss_ce_epoch), epoch)
            writer.add_scalar('loss/finetune_loss_clip_epoch', float(finetune_loss_clip_epoch), epoch)
            writer.add_scalar('lr/leaning_rate', lr_scheduler._get_lr(epoch)[0], epoch)

            val_loss, val_auc, val_metrics, _ = valid_finetune(model, ecg_model, text_encoder, tokenizer,
                                                            val_dataloader, epoch, device, args, config, writer,
                                                            train_dataset.label_name)
            writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
            writer.add_scalar('loss/val_auc_epoch', val_auc, epoch)

            if best_val_auc < val_auc:
                with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
                    f.write("Save best valid model.\n")
                best_val_auc = val_auc
                save_obj = {
                    'model': model.state_dict(),
                    'ecg_model': ecg_model.state_dict(),
                    'text_encoder': text_encoder.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                file_path = ("/home/user/tyy/project/ked/trained_model/checkpoints_finetune/finetune_cpsc_" +
                             str(config['finetune_sample_rate']) + ".pt")
                with open(file_path, "wb") as f:
                    torch.save(save_obj, f)

                print("Start testing")
                test_loss, test_auc, test_metrics, _ = valid_finetune(model, ecg_model, text_encoder, tokenizer,
                                                                   test_dataloader, epoch, device, args, config, writer,
                                                                   train_dataset.label_name)
                writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
                writer.add_scalar('loss/test_auc_epoch', test_auc, epoch)
                log_stats = {**{f'finetune_{k}': v for k, v in finetune_stats.items()},
                             'epoch': epoch, 'val_loss': val_loss.item(),
                             **{f'val_{k}': v for k, v in val_metrics.items()},
                             'test_loss': test_loss.item(),
                             **{f'test_{k}': v for k, v in test_metrics.items()},
                             }
                with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")
            else:
                log_stats = {**{f'finetune_{k}': v for k, v in finetune_stats.items()},
                             'epoch': epoch, 'val_loss': val_loss.item(),
                             **{f'val_{k}': v for k, v in val_metrics.items()},
                             }

                with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
                    f.write(json.dumps(log_stats) + "\n")

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('Finetune time {}'.format(total_time_str))
    else:
        print("Start testing")
        writer = SummaryWriter(os.path.join(args.finetune_output_dir, 'log'))
        with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
            f.write(config["finetune_purpose"] + "\n")
        # test(model, ecg_model, text_encoder, tokenizer, test_dataloader, device, args, config)
        test_loss, test_auc, test_metrics, confidence_result = valid_finetune(model, ecg_model, text_encoder, tokenizer,
                                                           test_dataloader, 0, device, args, config, writer,
                                                           train_dataset.label_name)
        writer.add_scalar('loss/test_loss_epoch', test_loss, 0)
        writer.add_scalar('loss/test_auc_epoch', test_auc, 0)
        log_stats = {'test_loss': test_loss.item(),
                     **{f'test_{k}': v for k, v in test_metrics.items()},
                     }
        with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")
            for item in confidence_result:
                f.write(str(item) + "\n")

def get_text_features(model,text_list,tokenizer,device,max_length):
    text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features

def test(model, ecg_encoder, text_encoder, tokenizer, data_loader, device, args, config):
    # text_list = ["Normal ECG", "First-degree atrioventricular block","Left bundle branch block",
    #                        "Right bundle branch block", "ST-segment depression", "ST-segment elevated"]
    # text_list = ["Normal ECG", "Conduction Disturbance(first degree AV block, left/Right bundle branch block)", "ST-T changes(ST-segment depression/elevated)"]
    if config['test_ICBEB_class_nums'] == 9:
        label_augment = \
            {   'normal ECG': 'Normal ECG.',
                'Atrial fibrillation': 'Atrial fibrillation',
                'first degree AV block': 'focus on the PR interval. A PR interval greater than 200 milliseconds indicates first-degree AV block. Look for a QRS duration <120ms and a narrow QRS complex in first-degree AV block. This electrocardiogram diagnosed:First-degree atrioventricular block.',
                'left bundle branch block': 'focus on leads V1 and V6. Look for a wide QRS complex (>120ms) with a slurred or notched R wave in V1 and a wide, deep S wave in V6. There will also be a delay in the onset of the QRS complex.This electrocardiogram diagnosed:Left bundle branch block.',
                'Right bundle branch block': 'focus on leads V1 and V2. Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in lead I and a broad R wave in leads V1 and V2. Additionally, there may be ST segment and T wave changes in leads V1-V3.This electrocardiogram diagnosed:Right bundle branch block.',
                "atrial premature complex": "Premature atrial contraction ",
                "ventricular premature complex": "Premature ventricular contraction ",
                'non-specific ST depression': 'focus on leads that correspond to the area of the heart suspected to be affected. Look for a downward sloping ST-segment that is at least 1 mm below the baseline and is present in at least two contiguous leads.This electrocardiogram diagnosed:ST-segment depression.',
                'non-specific ST elevation': 'focus on leads that correspond to the area of the heart affected. Look for a concave upward ST-segment elevation greater than 1mm above the baseline, with or without T-wave inversion. This electrocardiogram diagnosed:ST-segment elevated.'}
        text_list = label_augment.keys()
        dist_csv_col = ['metric', "Normal ECG","Atrial fibrillation", "First-degree atrioventricular block", "Left bundle branch block",
                        "Right bundle branch block","Premature atrial contraction","Premature ventricular contraction", "ST-segment depression", "ST-segment elevated", 'mean']
    elif config['test_ICBEB_class_nums'] == 6:
        label_augment = \
            {'normal ECG': 'Normal ECG.',
             'first degree AV block': 'focus on the PR interval. A PR interval greater than 200 milliseconds indicates first-degree AV block. Look for a QRS duration <120ms and a narrow QRS complex in first-degree AV block. This electrocardiogram diagnosed:First-degree atrioventricular block.',
             'left bundle branch block': 'focus on leads V1 and V6. Look for a wide QRS complex (>120ms) with a slurred or notched R wave in V1 and a wide, deep S wave in V6. There will also be a delay in the onset of the QRS complex.This electrocardiogram diagnosed:Left bundle branch block.',
             'Right bundle branch block': 'focus on leads V1 and V2. Look for a wide QRS complex (>0.12 seconds) with a slurred S wave in lead I and a broad R wave in leads V1 and V2. Additionally, there may be ST segment and T wave changes in leads V1-V3.This electrocardiogram diagnosed:Right bundle branch block.',
             'non-specific ST depression': 'focus on leads that correspond to the area of the heart suspected to be affected. Look for a downward sloping ST-segment that is at least 1 mm below the baseline and is present in at least two contiguous leads.This electrocardiogram diagnosed:ST-segment depression.',
             'non-specific ST elevation': 'focus on leads that correspond to the area of the heart affected. Look for a concave upward ST-segment elevation greater than 1mm above the baseline, with or without T-wave inversion. This electrocardiogram diagnosed:ST-segment elevated.'}
        text_list = label_augment.keys()
        dist_csv_col = ['metric', "Normal ECG", "First-degree atrioventricular block", "Left bundle branch block",
                        "Right bundle branch block", "ST-segment depression", "ST-segment elevated", 'mean']
    elif config['test_ICBEB_class_nums'] == 3:
        text_list = ["Normal ECG", "Conduction Disturbance(first degree AV block, left/Right bundle branch block)",
                     "ST-T changes(ST-segment depression/elevated)"]
        dist_csv_col = ['metric', "Normal ECG", "Conduction Disturbance", "ST-T changes", 'mean']

    save_result_path = os.path.join(args.output_dir, config["result_ICBEB_save_name"])
    f_result = open(save_result_path, 'w+', newline='')
    wf_result = csv.writer(f_result)
    wf_result.writerow(dist_csv_col)

    text_features = get_text_features(text_encoder, text_list, tokenizer, device, max_length=args.max_length)

    model.eval()
    ecg_encoder.eval()
    text_encoder.eval()

    gt = torch.FloatTensor()
    gt = gt.cuda()
    pred = torch.FloatTensor()
    pred = pred.cuda()

    for i, sample in enumerate(data_loader):
        signal = sample['signal'].to(device)
        if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
            signal = signal.unsqueeze(1)  # for lstm and resnet
        elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', "xresnet1d_101"]:
            signal = signal.transpose(1, 2)  # for lstm and resnet

        label = sample['label'].to(device)
        if config["use_ecgNet_Diagnosis"] == "ecgNet":
            label = torch.tensor(label, device=device)[:, [0, 2, 1]]
            zeros = torch.zeros(label.shape[0], 2, device=device)
            label = torch.cat((label[:, [0]], zeros[:,[0]], label[:, [1,2]], zeros[:,[1]]), dim=1)

        gt = torch.cat((gt, label), 0)

        with torch.no_grad():
            if config["ecg_model_name"] in ['resnet1d_wang', 'xresnet1d_101']:
                ecg_features = ecg_encoder(signal)
                ecg_features_pool = ecg_features.mean(-1)
            else:
                ecg_features, ecg_features_pool = ecg_encoder(signal)  # (32, 768, 300), (32, 768)

            if config["use_ecgNet_Diagnosis"] == "ecgNet":
                """"""
                pred = torch.cat((pred, ecg_features), 0)
            else:
                pred_class = model(ecg_features.transpose(1, 2), text_features)  # (64, 768, 14), (64, 3, 768)->(64,3,2)
                pred_class = torch.softmax(pred_class, dim=-1)
                pred = torch.cat((pred, pred_class[:,:,1]), 0)      # 取出是这个label的概率
    if not config["use_ecgNet_Diagnosis"] == "ecgNet":
        num_class = config['test_ICBEB_class_nums']
        AUROCs = compute_AUCs(gt.cpu().numpy(), pred.cpu().numpy(), n_class=num_class)
    else:
        num_class = 5
    mccs, threshold = compute_mccs(gt.cpu().numpy(), pred.cpu().numpy(), n_class=num_class)
    F1s = compute_F1s_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=num_class)
    Accs = compute_Accs_threshold(gt.cpu().numpy(), pred.cpu().numpy(), threshold, n_class=num_class)

    plot_confusion_matrix_final(gt.cpu().numpy(), pred.cpu().numpy(), threshold, num_class, list(text_list))


    output = []
    output.append('threshold')
    output.append(threshold)
    wf_result.writerow(output)
    if not config["use_ecgNet_Diagnosis"] == "ecgNet":
        wf_result.writerow(AUROCs)
    wf_result.writerow(F1s)
    wf_result.writerow(mccs)
    wf_result.writerow(Accs)

    AUROCs_list = []
    mccs_list = []
    F1s_list = []
    Accs_list = []

    data_len = len(gt)
    for idx in range(1000):
        randnum = random.randint(0, 5000)
        random.seed(randnum)
        gt_idx = random.choices(gt.cpu().numpy(), k=data_len)
        random.seed(randnum)
        pred_idx = random.choices(pred.cpu().numpy(), k=data_len)
        gt_idx = np.array(gt_idx)
        pred_idx = np.array(pred_idx)

        if not config["use_ecgNet_Diagnosis"] == "ecgNet":
            AUROCs_idx = compute_AUCs(gt_idx, pred_idx)
        mccs_idx, threshold_idx = compute_mccs(gt_idx, pred_idx, n_class=num_class)
        F1s_idx = compute_F1s_threshold(gt_idx, pred_idx, threshold_idx, n_class=num_class)
        Accs_idx = compute_Accs_threshold(gt_idx, pred_idx, threshold_idx, n_class=num_class)

        if not config["use_ecgNet_Diagnosis"] == "ecgNet":
            AUROCs_list.append(AUROCs_idx[1:])  # 1000,5
        mccs_list.append(mccs_idx[1:])
        F1s_list.append(F1s_idx[1:])
        Accs_list.append(Accs_idx[1:])
    if not config["use_ecgNet_Diagnosis"] == "ecgNet":
        AUROCs_5, AUROCs_95, AUROCs_mean = get_sort_eachclass(AUROCs_list, n_class=num_class)
        output = []
        output.append('perclass_AUROCs_5')
        output.extend(AUROCs_5)
        wf_result.writerow(output)
        output = []
        output.append('perclass_AUROCs_95')
        output.extend(AUROCs_95)
        wf_result.writerow(output)
        output = []
        output.append('perclass_AUROCs_mean')
        output.extend(AUROCs_mean)
        wf_result.writerow(output)

    mccs_5, mccs_95, mccs_mean = get_sort_eachclass(mccs_list, n_class=num_class)
    output = []
    output.append('perclass_mccs_5')
    output.extend(mccs_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_mccs_95')
    output.extend(mccs_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_mccs_mean')
    output.extend(mccs_mean)
    wf_result.writerow(output)

    F1s_5, F1s_95, F1s_mean = get_sort_eachclass(F1s_list, n_class=num_class)
    output = []
    output.append('perclass_F1s_5')
    output.extend(F1s_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_F1s_95')
    output.extend(F1s_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_F1s_mean')
    output.extend(F1s_mean)
    wf_result.writerow(output)

    Accs_5, Accs_95, Accs_mean = get_sort_eachclass(Accs_list, n_class=num_class)
    output = []
    output.append('perclass_Accs_5')
    output.extend(Accs_5)
    wf_result.writerow(output)
    output = []
    output.append('perclass_Accs_95')
    output.extend(Accs_95)
    wf_result.writerow(output)
    output = []
    output.append('perclass_Accs_mean')
    output.extend(Accs_mean)
    wf_result.writerow(output)
    f_result.close()

def plot_confusion_matrix_final(gt, pred, threshold, n_class, label_list):
    gt_np = gt
    pred_np = pred

    for i in range(n_class):
        pred_np[:, i][pred_np[:, i] >= threshold[i]] = 1
        pred_np[:, i][pred_np[:, i] < threshold[i]] = 0
    matrix = multilabel_confusion_matrix(gt_np, pred_np)
    plot_confusion_matrix(matrix, label_list)

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

# macro
def compute_AUCs(gt, pred, n_class=6):
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

def plot_confusion_matrix(matrix, labels):
    fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(8, 14))

    for i, (ax, matrix, label) in enumerate(zip(axs.flatten(), matrix, labels)):
        ax.matshow(matrix, cmap='coolwarm')
        ax.set(title=' {}'.format(label),
               xlabel='Predicted Label', ylabel='True Label',
               xticks=[0, 1], yticks=[0, 1])
        for (x, y), value in np.ndenumerate(matrix):
            ax.text(x, y, int(value), ha='center', va='center')

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    config = yaml.load(open('./configs/Res_train.yaml', 'r'), Loader=yaml.Loader)
    if config['finetune']:
        model_list = ['best_valid_all_increase_with_augment_epoch_3.pt']
    else:
        model_list = [
            # "best_valid_all_increase_zhipuai_augment_epoch_3.pt",
            "best_valid_all_increase_gemini_augment_epoch_3.pt"
            # 'best_valid_all_increase_with_augment_epoch_3.pt',
            # 'best_valid_all_base_no_augment_epoch_3.pt',
            # 'best_valid_all_increase_no_augment_epoch_3.pt',
            # 'best_valid_all_base_with_augment_epoch_3.pt',
        ]
    for model in model_list:
        if 'no_augment' in model:
            model_path = '/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv/' + model
        else:
            model_path = '/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv_copy/' + model
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='./configs/Res_train.yaml')
        parser.add_argument('--checkpoint', default=model_path)
        parser.add_argument('--ignore_index', default=False, type=bool)
        parser.add_argument('--bert_model_name', default='emilyalsentzer/Bio_ClinicalBERT')
        parser.add_argument('--output_dir', default='./output_test/')
        parser.add_argument('--finetune_output_dir', default='./output/output_finetune/cpsc')
        parser.add_argument('--max_length', default=256, type=int)
        parser.add_argument('--loss_ratio', default=1, type=int)
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--distributed', default=False, type=bool)
        parser.add_argument('--action', default='train')
        args = parser.parse_args()

        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

        torch.cuda.current_device()
        torch.cuda._initialized = True

        if config['finetune']:
            config["finetune_purpose"] = "######################" + model + "############" + str(config['finetune_sample_rate']) + "############"
        else:
            config["finetune_purpose"] = "######################" + model + "########################"
        main(args, config)