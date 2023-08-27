# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-07-24 14:28

import argparse
import os
import cv2
try:
    import ruamel.yaml as yaml
except:
    import ruamel_yaml as yaml
import numpy as np
import random
import json
import math
from skimage import io
from tqdm import tqdm
from pathlib import Path
from functools import partial
from einops import rearrange

from torchvision import transforms
from PIL import Image

import torch
import torch.backends.cudnn as cudnn
from transformers import AutoModel, BertConfig, AutoTokenizer
from models.clip_model import CLP_clinical, TQNModel
from models.resnet1d_wang import resnet1d_wang


def get_text_features(model, text_list, tokenizer, device, max_length):
    text_token = tokenizer(list(text_list), add_special_tokens=True, max_length=max_length, pad_to_max_length=True,
                           return_tensors='pt').to(device=device)
    text_features = model.encode_text(text_token)
    return text_features


def get_gt(syms_list, text_list):
    gt_class = np.zeros((10))
    gt = []
    for syms in syms_list:
        syms_class = text_list.index(syms)
        gt.append(syms_class)
        gt_class[syms_class] = 1
    return gt, gt_class

from dataset.ecgDataset import ICBEBDataset, FinetuneDataset
def get_ecg_feature_and_label(disease_name):
    """ventricular premature complex 或者 atrial premature complex"""
    X = np.load("../dataset/clinical_dataset/X_clinical_data.npy", allow_pickle=True)
    y = np.load("../dataset/clinical_dataset/y_clinical_data.npy", allow_pickle=True)
    # X = np.load("../dataset/shaoxing/signal_data.npy", allow_pickle=True)
    # y = np.load("../dataset/shaoxing/label_data.npy", allow_pickle=True)
    test_dataset = FinetuneDataset(X, y, "clinical")
    label_list = test_dataset.label_name
    label_index = label_list.index(disease_name)
    class_indices = np.argwhere(y[:, label_index] == 1).flatten()
    # 随机选择一个样本索引
    random_index = np.random.choice(class_indices)
    # 获取对应样本及其标签
    random_sample = X[random_index]
    random_label = y[random_index, label_index]
    print("样本索引为：",random_index)
    return torch.FloatTensor(random_sample.astype('float64')), disease_name, label_index, test_dataset.label_name, random_index



def main(args, disease_name):
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count())
    torch.set_default_tensor_type('torch.FloatTensor')

    # fix the seed for reproducibility
    seed = args.seed
    torch.manual_seed(seed)
    # np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    # model
    image_encoder = resnet1d_wang(num_classes=config['test_clinical_class_nums'], input_channels=12, kernel_size=5,
                          ps_head=0.5, lin_ftrs_head=[128], inplanes=768,use_ecgNet_Diagnosis=config["use_ecgNet_Diagnosis"]).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(args.bert_model_name, do_lower_case=True, local_files_only=True)
    text_encoder = CLP_clinical(bert_model_name=args.bert_model_name, freeze_layers=config['freeze_layers']).to(device)

    model = TQNModel(num_layers=config["tqn_model_layers"]).to(device)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    image_state_dict = checkpoint['ecg_model']
    image_encoder.load_state_dict(image_state_dict)
    text_state_dict = checkpoint['text_encoder']
    text_encoder.load_state_dict(text_state_dict)
    state_dict = checkpoint['model']
    model.load_state_dict(state_dict)

    model.eval()
    image_encoder.eval()
    text_encoder.eval()

    normalize = transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    transform = transforms.Compose([
        transforms.Resize([512, 512], interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # text_list = ['Atelectasis', 'Calcification', 'Consolidation', 'Effusion', 'Emphysema', 'Fibrosis', 'Fracture',
    #              'Mass', 'Nodule', 'Pneumothorax']
    text_list = ["Normal ECG",
                "Sinus arrhythmia",
                "Atrial Fibrillation",
                "atrial premature complex",
                "ventricular premature complex",
                "Left anterior fascicular block",
                "Supraventricular Tachycardia",
                "Atrial Flutter",
                "first degree AV block",
                "complete right bundle branch block"]

    text_features = get_text_features(text_encoder, text_list, tokenizer, device, max_length=args.max_length)
    # json_info = json.load(open(args.test_path, 'r'))

    gt = []  # label of each boxes
    gt_boxes = []
    gt_class = []  # 0-1label

    signal, report, label_index, text_list, random_index = get_ecg_feature_and_label(disease_name)
    gt_index, gt_class_index = get_gt([disease_name], text_list)

    # for data_index in tqdm(range(len(json_info))):
    #     json_index = json_info[data_index]
    #     file_name = json_index['file_name']
    #     syms_list = json_index['syms']
    #     # boxes_index = json_index['boxes']
    #     gt_index, gt_class_index = get_gt(disease_name, text_list)
    #     gt.append(gt_index)
    #     # gt_boxes.append(boxes_index)
    #     gt_class.append(gt_class_index)
    #     # print(gt_index,gt_class_index)
    #     data_path = os.path.join('./ChestX-Det10-Dataset/test_data', file_name)
    #     img = Image.open(data_path).convert('RGB')
    #     image = transform(img)
    #     image = image.unsqueeze(0).to(device)
    if (config["ecg_model_name"] == 'LSTM') or (config["ecg_model_name"] == 'resnet'):
        signal = signal.unsqueeze(1).to(device)  # for lstm and resnet
    elif config["ecg_model_name"] in ['ecgNet', 'resnet1d_wang', 'xresnet1d_101']:
        signal = signal.transpose(0, 1)  # for lstm and resnet
        signal = signal.unsqueeze(0).to(device)
    with torch.no_grad():

        image_features = image_encoder(signal)
        pred_class_index, atten_map_index = model(image_features.transpose(1, 2), text_features, return_atten=True)
        pred_class_index = torch.softmax(pred_class_index, dim=-1)
        atten_map_index = np.array(atten_map_index.cpu().numpy())

        if len(gt_index) == 0:
            pass
        else:
            for class_index in range(len(gt_class_index)):
                if gt_class_index[class_index] == 1:
                    # img_size = signal.squeeze().size
                    img_size = (1,1000)
                    save_attn_path = os.path.join(os.path.join(args.output_dir, 'visualize'),
                                                '_' + str(class_index) + '_atten.png')
                    # print(atten_map_index.shape,class_index)
                    atten_map = rearrange(atten_map_index[0][class_index], ' (w h) -> w h', w=250, h=1)
                    atten_map = 250 * atten_map / np.max(atten_map)
                    atten_map = cv2.resize(atten_map, img_size, interpolation=cv2.INTER_LINEAR).astype(np.uint8)
                    # atten_map = cv2.applyColorMap(atten_map, cv2.COLORMAP_JET)
                    atten_map = np.array(atten_map).squeeze()
                    atten_map = normalize_array(atten_map)
                    signal = signal.squeeze()
                    draw_heatmap(signal[1].cpu(), atten_map, random_index, disease_name)


def normalize_array(array):
    min_value = np.min(array)
    max_value = np.max(array)
    normalized = 2 * (array - min_value) / (max_value - min_value)
    return normalized

import matplotlib.pyplot as plt
def draw_heatmap(signal, cam, random_index, disease_name):
    fig, ax = plt.subplots(figsize=(12, 2), dpi=200)
    ax.set_xticks(np.arange(0, 10.5, 0.5))
    ax.set_yticks(np.arange(-5.0, +5.0, 0.5))
    ax.minorticks_on()
    ax.grid(which='major', linestyle='-', linewidth='0.5', color='red')
    ax.grid(which='minor', linestyle='-', linewidth='0.5', color=(1, 0.7, 0.7))
    #     ax.set_ylim(mean_value-5.0, mean_value+5.0)
    #     ax.set_xlim(0, 8.5)
    for i, p in enumerate([1]):
        sig = signal
        t = np.arange(0, len(sig) * 1 / 100, 1 / 100)
        ax.plot(t, sig, label="ECG signal", color='k')
    #      PVC==================================
    #     for [x1,x2] in [[0, 0.4]]:
    #         print(x1,x2)
    #         ax.plot(np.arange(x1*500,x2*500,1)/500, sig[int(x1*500):int(x2*500)],color='b')
    # ================pac===========================================
    #     for [x1,x2] in [[1, 1.8],[3.8,4.6],[5.3,6.1],[7.5,4096/500]]:
    #         print(x1,x2)
    #         ax.plot(np.arange(x1*500,x2*500,1)/500, sig[int(x1*500):int(x2*500)],color='b')

    ax.legend(loc='upper right', bbox_to_anchor=(.96, 1.0), prop={'family': 'Arial', 'size': 10})
    ax.set(xlabel='time (s)')
    ax.set(ylabel='Voltage (mV)')

    ax.autoscale(tight=True)
    ax.set_xlim(left=0, right=10.5)
    ax.set_ylim(top=1.5, bottom=-1.2)
    plt.xticks(fontproperties='Arial', size=10)
    plt.yticks(fontproperties='Arial', size=10)
    plt.xlabel('time (s)', fontdict={'family': 'Arial', 'size': 10})
    plt.ylabel('Voltage (mV)', fontdict={'family': 'Arial', 'size': 10})
    plt.title("("+disease_name+")sample index: "+str(random_index))
    x, y = np.meshgrid(np.linspace(-0, 10.5, 1000), np.linspace(0, 1.2, 1000 // 6))
    z = np.zeros((1000 // 6, 1050))
    for ii in range(1050):
        if ii < 1000:
            z[:, ii] = cam[ii]
        else:
            z[:, ii] = np.min(cam)
    #         YlOrRd
    plt.imshow(z, cmap=plt.cm.YlOrRd, alpha=0.5, interpolation='bilinear', origin="lower",
               extent=[0, 10.5, -1, 1.5], label='map')
    #     plt.colorbar()

    ax2 = ax.twinx()

    #         print(z)
    #     z = (1 - x / 2 + x**5 + y**3) * np.exp(-(x**2 + y**2))

    ax2.plot(t, cam, label='attention weight', color='b', linestyle='--', linewidth=1.5)
    ax2.set_ylim(top=1.2, bottom=-1)
    plt.yticks(fontproperties='Arial', size=10)
    ax2.set(ylabel='attention weight')
    plt.ylabel('attention weight', fontdict={'family': 'Arial', 'size': 10})
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/Res_train.yaml')
    parser.add_argument('--momentum', default=False, type=bool)
    parser.add_argument('--checkpoint', default='/home/tyy/unECG/trained_model/checkpoints/best_valid.pt')
    parser.add_argument('--freeze_bert', default=False, type=bool)
    parser.add_argument('--ignore_index', default=False, type=bool)
    parser.add_argument("--use_entity_features", action="store_true")
    parser.add_argument('--image_encoder_name', default='resnet')
    parser.add_argument('--bert_pretrained', default='')
    parser.add_argument('--bert_model_name', default='emilyalsentzer/Bio_ClinicalBERT')
    parser.add_argument('--save_result_path', default='./output/res_512/visualize')
    parser.add_argument('--output_dir', default='./output/res_512')
    parser.add_argument('--max_length', default=256, type=int)
    parser.add_argument('--loss_ratio', default=1, type=int)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', default='2')
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)
    Path(args.save_result_path).mkdir(parents=True, exist_ok=True)
    for i in range(15):
        main(args, "Atrial Flutter")