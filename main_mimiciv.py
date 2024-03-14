# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-06-03 15:14


import argparse
import logging
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml
import numpy as np
import random
import time
import datetime
import json
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torch.utils.data.distributed
import pandas as pd
from sklearn.model_selection import train_test_split

from tensorboardX import SummaryWriter
from transformers import AutoTokenizer

from factory import utils
from scheduler import create_scheduler
from optim import create_optimizer
from engine.train_fg import train,valid_on_ptb
from models.clip_model import CLP_clinical, ModelDense,TQNModel
from dataset.ecgDataset import NewECGDataset, TotalLabelDataset, MimicivDataset
# from dataset_new import NewECGDataset
from models.old_model.model_new import ResNet1D
from models.ECGNet import ECGNet
from models.resnet1d_wang import resnet1d_wang
from models.xresnet1d_101 import xresnet1d101
from models.cpc import CPCModel


import gc
import ast
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"




def main(args, config):
    """"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 释放 GPU 缓存
    torch.cuda.empty_cache()
    # 删除未使用对象
    gc.collect()
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']


    # /home/tyy/ecg_ptbxl or /home/user/tyy/project/ked/dataset/ptb-xl
    # with open("./dataset/mimiciv/data_y_total_train.json", "r") as f:
    #     X_train = json.load(f)
    # with open("./dataset/mimiciv/data_y_total_val.json", "r") as f:
    #     X_val = json.load(f)
    # with open("./dataset/mimiciv/data_y_total_test.json", "r") as f:
    #     X_test = json.load(f)
    X_train = pd.read_json("./dataset/mimiciv/data_y_total_train.json")
    X_val = pd.read_json("./dataset/mimiciv/data_y_total_val.json")
    X_test = pd.read_json("./dataset/mimiciv/data_y_total_test.json")
    y_train = np.load("./dataset/mimiciv/y_train_one_hot_data.npy", allow_pickle=True)
    y_test = np.load("./dataset/mimiciv/y_test_one_hot_data.npy", allow_pickle=True)
    y_val = np.load("./dataset/mimiciv/y_val_one_hot_data.npy", allow_pickle=True)



    # X_feature = pd.read_json('/home/tyy/project/ecgfm_ked/dataset/ptb-xl/ptb-xl-plus/train_sample_feature_desc_result.json')
    # X_report = pd.read_csv("/home/tyy/ecg_ptbxl/output/exp0/data/total_report_train_final.csv", index_col=[0])


    train_dataset = MimicivDataset(X_train, y_train, useAugment=config["use_report_augment"],
                                      use_what_label=config['use_what_label'], useFeature=config["use_feature_augment"],
                                   mimic_augment_type=config['mimic_augment_type'])
    # else:
    #     train_dataset = TotalLabelDataset(X_train, y_train, X_report, useAugment=config["use_report_augment"],
    #                                       use_what_label=config['use_what_label'])
    test_dataset = MimicivDataset(X_test, y_test, use_what_label=config['use_what_label'])
    val_dataset = MimicivDataset(X_val, y_val, use_what_label=config['use_what_label'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['batch_size'],
                                  num_workers=0,
                                  sampler=None,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=None)

    test_dataloader = DataLoader(test_dataset,
                                  batch_size=config['batch_size']*10,
                                  num_workers=0,
                                  sampler=None,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=None)
    val_dataloader = DataLoader(val_dataset,
                                 batch_size=config['batch_size']*10,
                                 num_workers=0,
                                 sampler=None,
                                 shuffle=True,
                                 drop_last=True,
                                 collate_fn=None)
    train_dataloader.num_samples = len(train_dataset)
    train_dataloader.num_batches = len(train_dataloader)
    val_dataloader.num_samples = len(val_dataset)
    val_dataloader.num_batches = len(val_dataloader)
    test_dataloader.num_samples = len(test_dataset)
    test_dataloader.num_batches = len(test_dataloader)


    if config["ecg_model_name"] == 'resnet':
        ecg_model = ResNet1D(in_channels=1, base_filters=768, kernel_size=1, stride=2, groups = config["batch_size"],
                             n_block = config["ecg_model_layers"], n_classes=config['class_num']).to(device=device)
    elif config["ecg_model_name"] == 'densenet':
        ecg_model = ModelDense(dense_base_model='densenet121').to(device=device)
    elif config["ecg_model_name"] == 'ecgNet':
        ecg_model = ECGNet(input_channel=1, num_classes=config["class_num"],use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'swinT':
        ecg_model = CPCModel(input_channels=12, strides=[2, 2, 2, 2], kss=[10, 4, 4, 4],
             features=[512] * 4, n_hidden=512, n_layers=2,
             mlp=False, lstm=True, bias_proj=False,
             num_classes=5, skip_encoder=False,
             bn_encoder=True,
             lin_ftrs_head=[512],
             ps_head=0.5,
             bn_head=True).to(device=device)
    elif config["ecg_model_name"] == 'resnet1d_wang':
        ecg_model = resnet1d_wang(num_classes=config["class_num"], input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[128], inplanes=768,use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    elif config["ecg_model_name"] == 'xresnet1d_101':
        ecg_model = xresnet1d101(num_classes=config["class_num"], input_channels=12, kernel_size=5,
                                 ps_head=0.5).to(device=device)

    tokenizer = AutoTokenizer.from_pretrained(config["bert_model_name"], do_lower_case=True, local_files_only=False)
    text_encoder = CLP_clinical(bert_model_name=config["bert_model_name"], freeze_layers=config['freeze_layers']).to(device=device)

    model = TQNModel(num_layers=config["tqn_model_layers"]).to(device=device)

    arg_opt = utils.AttrDict(config['optimizer'])
    optimizer = create_optimizer(arg_opt, model, ecg_model, text_encoder)
    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    print("Start training")
    start_time = time.time()
    writer = SummaryWriter(os.path.join(args.output_dir, 'log'))
    best_val_auc = 0.0

    with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
        f.write(config["purpose"] + "\n")
        f.write("batch size: "+str(config["batch_size"])+",freeze_bert: "+str(config["freeze_layers"])+",ecg_model_layers: "+
                str(config["ecg_model_layers"])+",TQN_model_layers: "+str(config["tqn_model_layers"])+
                ",temperature: "+str(config["temperature"]) + ", loss_type: "+str(config["loss_type"]) + ", uniCl_type: "
                + str(config["uniCl_type"])+ ", use_report_augment: " + str(config["use_report_augment"])+ ", use_ecgNet_Diagnosis: "
                + str(config["use_ecgNet_Diagnosis"]) + ", loss_ratio: " + str(config["loss_ratio"]) + "\n")

    for epoch in range(start_epoch, max_epoch):
        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)
        train_stats = train(model, ecg_model, text_encoder, tokenizer, train_dataloader, optimizer, epoch,
                            warmup_steps, device, lr_scheduler, args, config, writer)

        for k, v in train_stats.items():
            if k == 'loss':
                train_loss_epoch = v
            elif k == 'loss_ce':
                train_loss_ce_epoch = v
            elif k == 'loss_clip':
                train_loss_clip_epoch = v

        writer.add_scalar('loss/train_loss_epoch', float(train_loss_epoch), epoch)
        writer.add_scalar('loss/train_loss_ce_epoch', float(train_loss_ce_epoch), epoch)
        writer.add_scalar('loss/train_loss_clip_epoch', float(train_loss_clip_epoch), epoch)
        writer.add_scalar('lr/leaning_rate', lr_scheduler._get_lr(epoch)[0], epoch)

        val_loss, val_auc, val_metrics = valid_on_ptb(model, ecg_model, text_encoder, tokenizer,
                                                              val_dataloader, epoch, device, args, config, writer)
        writer.add_scalar('loss/val_loss_epoch', val_loss, epoch)
        writer.add_scalar('loss/val_auc_epoch', val_auc, epoch)

        if best_val_auc < val_auc:
            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
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
            if config['class_num'] == 5:
                save_file_name = "best_valid_5"
            elif config['class_num'] == 23:
                save_file_name = "best_valid_23"
            elif config['class_num'] == 44:
                save_file_name = "best_valid_44"
            elif config['class_num'] == 12:
                save_file_name = "best_valid_12"
            elif config['class_num'] == 19:
                save_file_name = "best_valid_19"
            else:
                """
                    best_valid_new 是仅使用feature 增强的
                    best_valid_all 是使用feature和report增强的
                """
                save_file_name = "best_valid_all_increase_zhipuai_augment_epoch_" + str(epoch)
            # /home/user/tyy/project/ked or /home/tyy/project/ecgfm_ked
            with open("/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv/" + save_file_name + ".pt", "wb") as f:
                torch.save(save_obj, f)

            test_loss, test_auc, test_metrics = valid_on_ptb(model, ecg_model, text_encoder, tokenizer,
                                                                     test_dataloader, epoch, device, args, config,
                                                                     writer)
            writer.add_scalar('loss/test_loss_epoch', test_loss, epoch)
            writer.add_scalar('loss/test_auc_epoch', test_auc, epoch)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                         'test_loss': test_loss.item(),
                         **{f'test_{k}': v for k, v in test_metrics.items()},
                         }

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")
        else:
            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                         'epoch': epoch, 'val_loss': val_loss.item(),
                         **{f'val_{k}': v for k, v in val_metrics.items()},
                         }

            with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
                f.write(json.dumps(log_stats) + "\n")

        if utils.is_main_process():
            save_obj = {
                'model': model.state_dict(),
                'ecg_model': ecg_model.state_dict(),
                'text_encoder': text_encoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }

            with open("/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv/checkpoint_state.pt", "wb") as f:
                torch.save(save_obj, f)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Res_train.yaml')
    parser.add_argument('--freeze_bert', default=True, type=bool)
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument('--output_dir', default='output_mimiciv')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--gpu', type=str,default='1', help='gpu')
    parser.add_argument('--distributed', default=False, type=bool)
    parser.add_argument('--action', default='train')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

    torch.cuda.current_device()
    torch.cuda._initialized = True

    logging.info("Params:")
    params_file = os.path.join(args.output_dir, "params.txt")
    with open(params_file, "w") as f:
        for name in sorted(vars(args)):
            val = getattr(args, name)
            logging.info(f"  {name}: {val}")
            f.write(f"{name}: {val}\n")
    main(args, config)