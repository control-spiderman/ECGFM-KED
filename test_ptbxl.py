# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-07-24 13:57
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tensorboardX import SummaryWriter

from factory import utils
from dataset.ecgDataset import FinetuneDataset
from models.ECGNet import ECGNet
from models.clip_model import CLP_clinical, TQNModel
from models.resnet1d_wang import resnet1d_wang
from models.xresnet1d_101 import xresnet1d101
from engine.finetune_fg import valid_finetune, finetune

import pandas as pd
import argparse
import os
import ruamel.yaml as yaml
import numpy as np
import random
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from optim import create_optimizer
from scheduler import create_scheduler

def down_sample_train_data(X_train, y_train, X_report, sample_rate=1.0):
    num_samples = int(len(X_train) * sample_rate)
    indices = np.arange(len(X_train))
    random_indices = np.random.choice(indices, size=num_samples, replace=False)
    X_train_new = X_train[random_indices]
    y_train_new = y_train[random_indices]
    X_report_new = X_report.iloc[random_indices]
    return X_train_new, y_train_new, X_report_new

def main(args, config):
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    if config["ptbxl_use_what_label"] == "diagnosis_label":   # 44类
        X_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/X_train.npy", allow_pickle = True)
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/X_test.npy", allow_pickle = True)
        y_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/y_train.npy", allow_pickle = True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/y_test.npy", allow_pickle = True)
        X_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/X_val.npy", allow_pickle = True)
        y_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1/data/y_val.npy", allow_pickle = True)
        X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/report_train_clean_final.csv", index_col=[0])
    elif config["ptbxl_use_what_label"] == "subdiagnosis_label":  # 23类
        X_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/X_train.npy", allow_pickle = True)
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/X_test.npy", allow_pickle = True)
        y_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/y_train.npy", allow_pickle = True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/y_test.npy", allow_pickle = True)
        X_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/X_val.npy", allow_pickle = True)
        y_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/y_val.npy", allow_pickle = True)
        X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/report_train_clean_final.csv", index_col=[0])
    elif config["ptbxl_use_what_label"] == "all": # 71类
        X_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/X_train.npy", allow_pickle=True)
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/X_test.npy", allow_pickle=True)
        y_train = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/y_train.npy", allow_pickle=True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/y_test.npy", allow_pickle=True)
        X_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/X_val.npy", allow_pickle=True)
        y_val = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/y_val.npy", allow_pickle=True)
        X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/total_report_train_final.csv", index_col=[0])
    elif config["ptbxl_use_what_label"] == "form": #
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp2/data/X_test.npy", allow_pickle=True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp2/data/y_test.npy", allow_pickle=True)
        X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp2/data/total_report_train_final.csv", index_col=[0])
    elif config["ptbxl_use_what_label"] == "rhythm": #
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp3/data/X_test.npy", allow_pickle=True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp3/data/y_test.npy", allow_pickle=True)
        X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp3/data/total_report_train_final.csv", index_col=[0])
    else:
        X_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1/data/X_test.npy", allow_pickle=True)
        y_test = np.load("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp1.1.1/data/y_test.npy", allow_pickle=True)

    X_train, y_train, X_report = down_sample_train_data(X_train, y_train ,X_report, config['finetune_sample_rate'])
    # X_report = pd.read_csv("/home/user/tyy/project/ked/dataset/ptb-xl/output/exp0/data/total_report_train_final.csv", index_col=[0])
    train_dataset = FinetuneDataset(X_train, y_train, "ptb-xl", X_report, label_type=config["ptbxl_use_what_label"], isFinetune=config['finetune'])
    train_dataloader = DataLoader(train_dataset,
                                  batch_size=config['finetune_batch_size'],
                                  num_workers=0,
                                  sampler=None,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=None)
    val_dataset = FinetuneDataset(X_val, y_val, "ptb-xl", label_type=config["ptbxl_use_what_label"])
    val_dataloader = DataLoader(val_dataset,
                                batch_size=config['finetune_batch_size'],
                                num_workers=0,
                                sampler=None,
                                shuffle=True,
                                drop_last=True,
                                collate_fn=None)
    test_dataset = FinetuneDataset(X_test, y_test, "ptb-xl", label_type=config["ptbxl_use_what_label"],
                                   zeroshot_report_type=config['zeroshot_report_type'])
    test_dataloader = DataLoader(test_dataset,
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
        ecg_model = resnet1d_wang(num_classes=config['class_num'], input_channels=12, kernel_size=5,
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
        # start_time = time.time()
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
            f.write(config["finetune_purpose"] + config['ptbxl_use_what_label'] + "\n")
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
                file_path = "/home/user/tyy/project/ked/trained_model/checkpoints_finetune/finetune_ptbxl_" + str(
                    config['finetune_sample_rate']) + ".pt"
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

        # total_time = time.time() - start_time
        # total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        # print('Finetune time {}'.format(total_time_str))
    else:
        print("Start testing")
        writer = SummaryWriter(os.path.join(args.output_dir, 'log'))
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(config["finetune_purpose"] + config['ptbxl_use_what_label'] + "\n")
        # test(model, ecg_model, text_encoder, tokenizer, test_dataloader, device, args, config)
        test_loss, test_auc, test_metrics, _ = valid_finetune(model, ecg_model, text_encoder, tokenizer,
                                                           test_dataloader, 0, device, args, config, writer,
                                                           test_dataset.label_name)
        writer.add_scalar('loss/test_loss_epoch', test_loss, 0)
        writer.add_scalar('loss/test_auc_epoch', test_auc, 0)
        log_stats = {'test_loss': test_loss.item(),
                     **{f'test_{k}': v for k, v in test_metrics.items()},
                     }
        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(json.dumps(log_stats) + "\n")


if __name__ == '__main__':
    config = yaml.load(open('./configs/Res_train.yaml', 'r'), Loader=yaml.Loader)
    if config['finetune']:
        model_list = ['best_valid_all_increase_with_augment_epoch_3.pt']
    else:
        model_list = [
            # "best_valid_all_increase_zhipuai_augment_epoch_3.pt",
            # "best_valid_all_increase_gemini_augment_epoch_3.pt"
            'best_valid_all_increase_with_augment_epoch_3.pt',
            # 'best_valid_all_base_no_augment_epoch_3.pt',
            # 'best_valid_all_increase_no_augment_epoch_3.pt',
            # 'best_valid_all_base_with_augment_epoch_3.pt',
        ]
    for model in model_list:
        if 'no_augment' in model or 'zhipuai' in model:
            model_path = '/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv/' + model
        else:
            model_path = '/home/user/tyy/project/ked/trained_model/checkpoints_mimiciv_copy/' + model
        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default='./configs/Res_train.yaml')
        parser.add_argument('--checkpoint', default=model_path)
        parser.add_argument('--ignore_index', default=False, type=bool)
        parser.add_argument('--bert_model_name',
                            default='emilyalsentzer/Bio_ClinicalBERT')  # microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext or emilyalsentzer/Bio_ClinicalBERT
        parser.add_argument('--output_dir', default='./output_test/ptbxl')
        parser.add_argument('--finetune_output_dir', default='./output/output_finetune/ptbxl')
        parser.add_argument('--max_length', default=256, type=int)
        parser.add_argument('--loss_ratio', default=1, type=int)
        parser.add_argument('--device', default='cuda')
        parser.add_argument('--seed', default=42, type=int)
        parser.add_argument('--gpu', type=str, default='0', help='gpu')
        parser.add_argument('--distributed', default=False, type=bool)
        parser.add_argument('--action', default='train')
        args = parser.parse_args()


        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

        yaml.dump(config, open(os.path.join(args.output_dir, 'config.yaml'), 'w'))

        torch.cuda.current_device()
        torch.cuda._initialized = True
        if config['finetune']:
            config["finetune_purpose"] = ("######################" + model + "############"+
                                          str(config['finetune_sample_rate'])+"############")
        else:
            config["finetune_purpose"] = "######################" + model + "#####"+str(config['zeroshot_report_type'])+"###################"

        main(args, config)