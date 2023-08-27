# --coding:utf-8--
# project:
# user:User
# Author: tyy
# createtime: 2023-08-11 16:08

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from typing import Iterable

def _conv1d(in_planes,out_planes,kernel_size=3, stride=1, dilation=1, act="relu", bn=True, drop_p=0):
    lst=[]
    if(drop_p>0):
        lst.append(nn.Dropout(drop_p))
    lst.append(nn.Conv1d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=(kernel_size-1)//2, dilation=dilation, bias=not(bn)))
    if(bn):
        lst.append(nn.BatchNorm1d(out_planes))
    if(act=="relu"):
        lst.append(nn.ReLU(True))
    if(act=="elu"):
        lst.append(nn.ELU(True))
    if(act=="prelu"):
        lst.append(nn.PReLU(True))
    return nn.Sequential(*lst)

def listify(p=None, q=None):
    "Make `p` listy and the same length as `q`."
    if p is None: p=[]
    elif isinstance(p, str):          p = [p]
    elif not isinstance(p, Iterable): p = [p]
    #Rank 0 tensors in PyTorch are Iterable but don't have a length.
    else:
        try: a = len(p)
        except: p = [p]
    n = q if type(q)==int else len(p) if q is None else len(q)
    if len(p)==1: p = p * n
    assert len(p)==n, f'List len mismatch ({len(p)} vs {n})'
    return list(p)


def bn_drop_lin(n_in, n_out, bn=True, p=0., actn=None):
    "Sequence of batchnorm (if `bn`), dropout (with `p`) and linear (`n_in`,`n_out`) layers followed by `actn`."
    layers = [nn.BatchNorm1d(n_in)] if bn else []
    if p != 0: layers.append(nn.Dropout(p))
    layers.append(nn.Linear(n_in, n_out))
    if actn is not None: layers.append(actn)
    return layers

class CPCEncoder(nn.Sequential):
    'CPC Encoder'

    def __init__(self, input_channels, strides=[5, 4, 2, 2, 2], kss=[10, 8, 4, 4, 4], features=[512, 512, 512, 512],
                 bn=False):
        assert (len(strides) == len(kss) and len(strides) == len(features))
        lst = []
        for i, (s, k, f) in enumerate(zip(strides, kss, features)):
            lst.append(_conv1d(input_channels if i == 0 else features[i - 1], f, kernel_size=k, stride=s, bn=bn))
        super().__init__(*lst)
        self.downsampling_factor = np.prod(strides)
        self.output_dim = features[-1]
        # output: bs, output_dim, seq//downsampling_factor

    def encode(self, input):
        # bs = input.size()[0]
        # ch = input.size()[1]
        # seq = input.size()[2]
        # segments = seq//self.downsampling_factor
        # input_encoded = self.forward(input[:,:,:segments*self.downsampling_factor]).transpose(1,2) #bs, seq//downsampling, encoder_output_dim (standard ordering for batch_first RNNs)
        input_encoded = self.forward(input).transpose(1, 2)
        return input_encoded


# Cell
class CPCModel(nn.Module):
    "CPC model"

    def __init__(self, input_channels, strides=[5, 4, 2, 2, 2], kss=[10, 8, 4, 4, 4], features=[512, 512, 512, 512],
                 bn_encoder=False, n_hidden=512, n_layers=2, mlp=False, lstm=True, bias_proj=False, num_classes=None,
                 concat_pooling=True, ps_head=0.5, lin_ftrs_head=[512], bn_head=True, skip_encoder=False):
        super().__init__()
        assert (skip_encoder is False or num_classes is not None)  # pretraining only with encoder
        self.encoder = CPCEncoder(input_channels, strides=strides, kss=kss, features=features,
                                  bn=bn_encoder) if skip_encoder is False else None
        self.encoder_output_dim = self.encoder.output_dim if skip_encoder is False else None
        self.encoder_downsampling_factor = self.encoder.downsampling_factor if skip_encoder is False else None
        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.mlp = mlp

        self.num_classes = num_classes
        self.concat_pooling = concat_pooling

        self.rnn = nn.LSTM(self.encoder_output_dim if skip_encoder is False else input_channels, n_hidden,
                           num_layers=n_layers, batch_first=True) if lstm is True else nn.GRU(self.encoder.output_dim,
                                                                                              n_hidden,
                                                                                              num_layers=n_layers,
                                                                                              batch_first=True)

        if (num_classes is None):  # pretraining
            if (mlp):  # additional hidden layer as in simclr
                self.proj = nn.Sequential(nn.Linear(n_hidden, n_hidden), nn.ReLU(inplace=True),
                                          nn.Linear(n_hidden, self.encoder_output_dim, bias=bias_proj))
            else:
                self.proj = nn.Linear(n_hidden, self.encoder_output_dim, bias=bias_proj)
        else:  # classifier
            # slightly adapted from RNN1d
            layers_head = []
            if (self.concat_pooling):
                layers_head.append(AdaptiveConcatPoolRNN())

            # classifier
            nf = 3 * n_hidden if concat_pooling else n_hidden
            lin_ftrs_head = [nf, num_classes] if lin_ftrs_head is None else [nf] + lin_ftrs_head + [num_classes]
            ps_head = listify(ps_head)
            if len(ps_head) == 1:
                ps_head = [ps_head[0] / 2] * (len(lin_ftrs_head) - 2) + ps_head
            actns = [nn.ReLU(inplace=True)] * (len(lin_ftrs_head) - 2) + [None]

            for ni, no, p, actn in zip(lin_ftrs_head[:-1], lin_ftrs_head[1:], ps_head, actns):
                layers_head += bn_drop_lin(ni, no, bn_head, p, actn)
            self.head = nn.Sequential(*layers_head)

    def forward(self, input):
        # input shape bs,ch,seq
        if (self.encoder is not None):
            input_encoded = self.encoder.encode(input)
        else:
            input_encoded = input.transpose(1, 2)  # bs, seq, channels
        output_rnn, _ = self.rnn(input_encoded)  # output_rnn: bs, seq, n_hidden
        if (self.num_classes is None):  # pretraining
            return input_encoded, self.proj(output_rnn)
        else:  # classifier
            output = output_rnn.transpose(1, 2)  # bs,n_hidden,seq (i.e. standard CNN channel ordering)
            if (self.concat_pooling is False):
                output = output[:, :, -1]
            return self.head(output)

    def get_layer_groups(self):
        return (self.encoder, self.rnn, self.head)

    def get_output_layer(self):
        return self.head[-1]

    def set_output_layer(self, x):
        self.head[-1] = x

    def cpc_loss(self, input, target=None, steps_predicted=5, n_false_negatives=9, negatives_from_same_seq_only=False,
                 eval_acc=False):
        assert (self.num_classes is None)

        input_encoded, output = self.forward(input)  # input_encoded: bs, seq, features; output: bs,seq,features
        input_encoded_flat = input_encoded.reshape(-1, input_encoded.size(2))  # for negatives below: -1, features

        bs = input_encoded.size()[0]
        seq = input_encoded.size()[1]

        loss = torch.tensor(0, dtype=torch.float32).to(input.device)
        tp_cnt = torch.tensor(0, dtype=torch.int64).to(input.device)

        for i in range(input_encoded.size()[1] - steps_predicted):
            positives = input_encoded[:, i + steps_predicted].unsqueeze(1)  # bs,1,encoder_output_dim
            if (negatives_from_same_seq_only):
                idxs = torch.randint(0, (seq - 1), (bs * n_false_negatives,)).to(input.device)
            else:  # negative from everywhere
                idxs = torch.randint(0, bs * (seq - 1), (bs * n_false_negatives,)).to(input.device)
            idxs_seq = torch.remainder(idxs, seq - 1)  # bs*false_neg
            idxs_seq2 = idxs_seq * (idxs_seq < (i + steps_predicted)).long() + (idxs_seq + 1) * (
                        idxs_seq >= (i + steps_predicted)).long()  # bs*false_neg
            if (negatives_from_same_seq_only):
                idxs_batch = torch.arange(0, bs).repeat_interleave(n_false_negatives).to(input.device)
            else:
                idxs_batch = idxs // (seq - 1)
            idxs2_flat = idxs_batch * seq + idxs_seq2  # for negatives from everywhere: this skips step i+steps_predicted from the other sequences as well for simplicity

            negatives = input_encoded_flat[idxs2_flat].view(bs, n_false_negatives,
                                                            -1)  # bs*false_neg, encoder_output_dim
            candidates = torch.cat([positives, negatives], dim=1)  # bs,false_neg+1,encoder_output_dim
            preds = torch.sum(output[:, i].unsqueeze(1) * candidates, dim=-1)  # bs,(false_neg+1)
            targs = torch.zeros(bs, dtype=torch.int64).to(input.device)

            if (eval_acc):
                preds_argmax = torch.argmax(preds, dim=-1)
                tp_cnt += torch.sum(preds_argmax == targs)

            loss += F.cross_entropy(preds, targs)
        if (eval_acc):
            return loss, tp_cnt.float() / bs / (input_encoded.size()[1] - steps_predicted)
        else:
            return loss


# copied from RNN1d
class AdaptiveConcatPoolRNN(nn.Module):
    def __init__(self, bidirectional=False):
        super().__init__()
        self.bidirectional = bidirectional

    def forward(self, x):
        # input shape bs, ch, ts
        t1 = nn.AdaptiveAvgPool1d(1)(x)
        t2 = nn.AdaptiveMaxPool1d(1)(x)

        if (self.bidirectional is False):
            t3 = x[:, :, -1]
        else:
            channels = x.size()[1]
            t3 = torch.cat([x[:, :channels, -1], x[:, channels:, 0]], 1)
        out = torch.cat([t1.squeeze(-1), t2.squeeze(-1), t3], 1)  # output shape bs, 3*ch
        return out


if __name__ == '__main__':
    model = CPCModel(input_channels=12, strides=[2, 2, 2, 2], kss=[10, 4, 4, 4],
                     features=[512] * 4, n_hidden=512, n_layers=2,
                     mlp=False, lstm=True, bias_proj=False,
                     num_classes=5, skip_encoder=False,
                     bn_encoder=True,
                     lin_ftrs_head=[512],
                     ps_head=0.5,
                     bn_head=True)
    x = torch.rand(64, 12, 1000)
    result = model(x)
    print(result)