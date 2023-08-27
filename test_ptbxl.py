# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-07-24 13:57
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from transformers import AutoModel,BertConfig,AutoTokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix
from tensorboardX import SummaryWriter

from factory import utils
from dataset.ecgDataset import ICBEBDataset, FinetuneDataset
from models.model_new import ResNet1D
from models.ECGNet import ECGNet
from models.clip_model import CLP_clinical,ModelRes,ModelDense,TQNModel
from models.resnet1d_wang import resnet1d_wang
from models.xresnet1d_101 import xresnet1d101
from engine.finetune_fg import finetune, valid_finetune

from optim import create_optimizer
from scheduler import create_scheduler
import pandas as pd
import argparse
import os
import logging
import ruamel.yaml as yaml
import numpy as np
import random
import time
import datetime
import json
import csv
import math
from pathlib import Path
from functools import partial
from sklearn.metrics import roc_auc_score,matthews_corrcoef,f1_score,accuracy_score
import pickle

def down_sample_train_data(X_train, y_train, sample_rate=1.0):
    num_samples = int(len(X_train) * sample_rate)
    indices = np.arange(len(X_train))
    random_indices = np.random.choice(indices, size=num_samples, replace=False)
    X_train_new = X_train[random_indices]
    y_train_new = y_train[random_indices]
    return X_train_new, y_train_new

def main(args, config):
    device=torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    X_test = np.load("/home/tyy/ecg_ptbxl/output/exp0/data/X_test.npy", allow_pickle=True)
    y_test = np.load("/home/tyy/ecg_ptbxl/output/exp0/data/y_test.npy", allow_pickle=True)
    X_report = pd.read_csv("/home/tyy/ecg_ptbxl/output/exp0/data/total_report_train_final.csv", index_col=[0])
    test_dataset = FinetuneDataset(X_test, y_test, "clinical")
    test_dataloader = DataLoader(test_dataset,
                                 batch_size=config['finetune_batch_size'],
                                 num_workers=0,
                                 sampler=None,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=None)
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader)
    if config["ecg_model_name"] == 'resnet':
        # image_encoder = ModelRes(res_base_model='resnet50').to(device)
        ecg_model = ResNet1D(in_channels=1, base_filters=768, kernel_size=1, stride=2, groups=config["finetune_batch_size"],
                             n_block=config["ecg_model_layers"], n_classes=config['test_georgia_class_nums']).to(device=device)
    elif config["ecg_model_name"] == 'ecgNet':
        ecg_model = ECGNet(input_channel=1, use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'resnet1d_wang':
        ecg_model = resnet1d_wang(num_classes=config['test_georgia_class_nums'], input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[128], inplanes=768,use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'xresnet1d_101':
        ecg_model = xresnet1d101(num_classes=config["test_georgia_class_nums"], input_channels=12, kernel_size=5,
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

    print("Start testing")
    writer = SummaryWriter(os.path.join(args.finetune_output_dir, 'log'))
    with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
        f.write(config["finetune_purpose"] + "\n")
    # test(model, ecg_model, text_encoder, tokenizer, test_dataloader, device, args, config)
    test_loss, test_auc, test_metrics = valid_finetune(model, ecg_model, text_encoder, tokenizer,
                                                       test_dataloader, 0, device, args, config, writer,
                                                       test_dataset.label_name)
    writer.add_scalar('loss/test_loss_epoch', test_loss, 0)
    writer.add_scalar('loss/test_auc_epoch', test_auc, 0)
    log_stats = {'test_loss': test_loss.item(),
                 **{f'test_{k}': v for k, v in test_metrics.items()},
                 }
    with open(os.path.join(args.finetune_output_dir, "log.txt"), "a") as f:
        f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/Res_train.yaml')
    parser.add_argument('--checkpoint',
                        default='/home/tyy/unECG/trained_model/checkpoints/best_valid.pt')
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument('--bert_model_name',
                        default='emilyalsentzer/Bio_ClinicalBERT')  # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext or emilyalsentzer/Bio_ClinicalBERT
    parser.add_argument('--output_dir', default='./output_test/clinical')
    parser.add_argument('--finetune_output_dir', default='./output_finetune/ptbxl')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=str, default='0', help='gpu')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--action', default='train')
    parser.add_argument('--result_save_name', default='result_65_label_augment_new.csv')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    torch.cuda.current_device()
    torch.cuda._initialized = True

    purpose = ['zero-shot无报告增强', 'zero-shot 无报告增强且unicl', 'zero-shot unicl']
    checkpoint_model = ['best_valid_noAugment.pt', 'best_valid_noAugment_unicl.pt', 'best_valid_unicl.pt']
    main(args, config)