# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-04-20 13:56
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel,BertConfig,AutoTokenizer
# from pytorch_pretrained_vit import ViT

from models.transformer_decoder import *
from factory.visualization import visualization_tsne

from einops import rearrange
from typing import Tuple, Union, Callable, Optional
import numpy as np


class CLP_clinical(nn.Module):
    def __init__(self,
                 bert_model_name: str,
                 embed_dim: int = 768,
                 # freeze_layers: Union[Tuple[int, int], int] = None,
                 freeze_layers = [0,1]):
        super().__init__()
        self.bert_model = self._get_bert_basemodel(bert_model_name=bert_model_name, freeze_layers=freeze_layers)
        self.mlp_embed = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        self.embed_dim = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        self.init_parameters()

    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        for m in self.mlp_embed:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=self.embed_dim ** -0.5)

    def _get_bert_basemodel(self, bert_model_name, freeze_layers=[0,1]):  # 12
        try:
            print(bert_model_name)
            config = BertConfig.from_pretrained(bert_model_name, output_hidden_states=True)  # bert-base-uncased
            model = AutoModel.from_pretrained(bert_model_name, config=config)  # , return_dict=True)
            print("Text feature extractor:", bert_model_name)
            print("bert encoder layers:", len(model.encoder.layer))
        except:
            raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

        if freeze_layers:   # [0,1,2,3,4,5,6,7,8,9,10]
            for layer_idx in freeze_layers:
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = False
            for layer_idx in set(range(len(model.encoder.layer))) - set(freeze_layers):
                for param in list(model.encoder.layer[layer_idx].parameters()):
                    param.requires_grad = True
        # if freeze_layers != -1:
        #     for layer_idx in range(
        #             len(model.encoder.layer) - freeze_layers):  # Freeze all layers except last freeze_layers layers
        #         for param in list(model.encoder.layer[layer_idx].parameters()):
        #             param.requires_grad = False
        return model

    def encode_text(self, text):
        # input batch_size,token, return batch_size,dim
        output = self.bert_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        last_hidden_state, pooler_output, hidden_states = output[0], output[1], output[2]
        encode_out = self.mlp_embed(pooler_output)
        return encode_out

    def encode_origin_text(self, text):
        with torch.no_grad():
            output = self.bert_model(input_ids=text['input_ids'], attention_mask=text['attention_mask'])
        return output[0]

    # 代码中好像没用forward函数，而是都用的encode_text来获取编码，因此可学习参数还有三层mlp
    def forward(self, text1, text2):
        text1_features = self.encode_text(text1)
        text2_features = self.encode_text(text2)
        text1_features = F.normalize(text1_features, dim=-1)
        text2_features = F.normalize(text2_features, dim=-1)
        return text1_features, text2_features, self.logit_scale.exp()


class ModelRes(nn.Module):
    def __init__(self, res_base_model):
        super(ModelRes, self).__init__()
        self.resnet_dict = {"resnet50": models.resnet50(weights=True)}
                            # "resnet50": models.resnet50(pretrained=True)}
        self.resnet = self._get_res_basemodel(res_base_model)
        num_ftrs = int(self.resnet.fc.in_features/2)
        self.res_features = nn.Sequential(*list(self.resnet.children())[:-3])
        self.res_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.res_l2 = nn.Linear(num_ftrs, 768)


    def _get_res_basemodel(self, res_model_name):
        try:
            res_model = self.resnet_dict[res_model_name]
            print("Image feature extractor:", res_model_name)
            return res_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: resnet18 or resnet50")

    def forward(self, img):
        #return (batchsize, patch_num, dim)
        batch_size = img.shape[0]
        res_fea = self.res_features(img)     # (32,3,224,224)→(32,1024,14,14)
        res_fea = rearrange(res_fea,'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(res_fea,'b n d -> (b n) d')
        x = self.res_l1(h)
        x = F.relu(x)
        x = self.res_l2(x)
        out_emb = rearrange(x,'(b n) d -> b n d',b=batch_size)
        out_pool = torch.mean(out_emb,dim=1)
        return out_emb,out_pool


class ModelDense(nn.Module):
    def __init__(self, dense_base_model):
        super(ModelDense, self).__init__()

        self.densenet_dict = {"densenet121": models.densenet121(pretrained=True)}  # ,
        # "densenet161": models.densenet161(pretrained=True)}
        self.densenet = self._get_dense_basemodel(dense_base_model)
        num_ftrs = int(self.densenet.classifier.in_features)
        self.dense_features = self.densenet.features
        self.dense_l1 = nn.Linear(num_ftrs, num_ftrs)
        self.dense_l2 = nn.Linear(num_ftrs, 768)

    def _get_dense_basemodel(self, dense_base_model):
        try:
            dense_model = self.densenet_dict[dense_base_model]
            print("Image feature extractor:", dense_base_model)
            return dense_model
        except:
            raise ("Invalid model name. Check the config file and pass one of: densenet121 or densenet161")

    def forward(self, img):
        batch_size = img.shape[0]
        dense_fea = self.dense_features(img)  # N, 1024, 7,7
        dense_fea = rearrange(dense_fea, 'b d n1 n2 -> b (n1 n2) d')
        h = rearrange(dense_fea, 'b n d -> (b n) d')
        x = self.dense_l1(h)
        x = F.relu(x)
        x = self.dense_l2(x)
        out_emb = rearrange(x, '(b n) d -> b n d', b=batch_size)
        out_pool = torch.mean(out_emb, dim=1)
        return out_emb, out_pool

class TQNModel(nn.Module):
    def __init__(self,
                 embed_dim: int = 768,
                 class_num: int = 2,
                 num_layers: int = 3
                 ):
        super().__init__()
        self.d_model = embed_dim
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        decoder_layer = TransformerDecoderLayer(self.d_model, 4, 1024,
                                                0.1, 'relu', normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, num_layers, self.decoder_norm,
                                          return_intermediate=False)
        self.dropout_feas = nn.Dropout(0.1)

        self.mlp_head = nn.Sequential(  # nn.LayerNorm(768),
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Linear(2048, class_num)
        )
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def forward(self, ecg_features, text_features, return_atten=False): #（64，250，768） （5，768）
        # image_features (batch_size,patch_num,dim)
        # text_features (query_num,dim)
        batch_size = ecg_features.shape[0]
        ecg_features = ecg_features.transpose(0, 1)     # (250, 64, 768)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)     # (5,64,768)
        ecg_features = self.decoder_norm(ecg_features)
        text_features = self.decoder_norm(text_features)
        features, atten_map = self.decoder(text_features, ecg_features,
                                memory_key_padding_mask=None, pos=None, query_pos=None)     # (40, 32, 768)
        features = self.dropout_feas(features).transpose(0, 1)  # (32,5,768)
        out = self.mlp_head(features)  # (32,5,2)  做二分类，判断是还是不是这个类别
        if return_atten:
            return out, atten_map
        else:
            return out

    def visual_tsne(self, ecg_features, text_features, label_list):
        batch_size = ecg_features.shape[0]
        ecg_features = ecg_features.transpose(0, 1)  # (250, 64, 768)
        text_features = text_features.unsqueeze(1).repeat(1, batch_size, 1)  # (5,64,768)
        ecg_features = self.decoder_norm(ecg_features)
        text_features = self.decoder_norm(text_features)
        features = self.decoder(text_features, ecg_features,
                                memory_key_padding_mask=None, pos=None, query_pos=None)  # (40, 32, 768)
        features = self.dropout_feas(features).transpose(0, 1)  # (32,5,768)
        # features = features.max(1) # (32,768)
        # features = torch.nn.functional.max_pool1d(features, kernel_size=features.shape[1])

        m = nn.MaxPool1d(features.shape[1])
        features = m(features.transpose(1,2)).squeeze()
        visualization_tsne(features,label_list)


if __name__ == "__main__":
    # torch 1.10.2 to torch 1.12.1
    # torchvision-0.11.3 to torchvision-0.13.1

    image = torch.randn(1, 3, 224, 224)
    image_encoder = ModelRes(res_base_model='resnet50')
    # image_encoder = ModelDense(dense_base_model = 'densenet121')
    image_encoder(image)