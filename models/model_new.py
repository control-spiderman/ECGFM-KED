import torch
import torch.nn as nn
from block_new import get_sinusoid_encoding_table, get_attn_key_pad_mask, get_non_pad_mask, \
    get_subsequent_mask, EncoderLayer
from config import *
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
from torch.nn.modules.batchnorm import BatchNorm1d


class Encoder(nn.Module):


    def __init__(
            self,
            d_feature,
            n_layers, n_head, d_k, d_v,
            d_model, d_inner, dropout=0.1):
        super().__init__()

        n_position = d_feature + 1
        self.src_word_emb = nn.Conv1d(1, d_model, kernel_size=KS, padding=int((KS - 1) / 2))

        self.position_enc = nn.Embedding.from_pretrained(
            get_sinusoid_encoding_table(n_position, d_model, padding_idx=0),
            freeze=True)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model, d_inner, n_head, d_k, d_v, dropout=dropout)
            for _ in range(n_layers)])

    def forward(self, src_seq, src_pos):

        non_pad_mask = get_non_pad_mask(src_seq)
        slf_attn_mask = get_attn_key_pad_mask(seq_k=src_seq, seq_q=src_seq)
        enc_output = src_seq.unsqueeze(1)
        enc_output = self.src_word_emb(enc_output)
        enc_output = enc_output.transpose(1, 2)
        enc_output.add_(self.position_enc(src_pos))

        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output,
                non_pad_mask=non_pad_mask,
                slf_attn_mask=slf_attn_mask)
        return enc_output,


class Transformer(nn.Module):
    ''' A sequence to sequence model with attention mechanism. '''
    def __init__(
            self, device,
            d_feature, d_model, d_inner,
            n_layers, n_head, d_k=64, d_v=64, dropout = 0.1,
            class_num=2):

        super().__init__()

        self.encoder = Encoder(d_feature, n_layers, n_head, d_k, d_v, d_model, d_inner, dropout)
        self.device = device

        self.linear1_cov = nn.Conv1d(d_feature, 1, kernel_size=1)
        self.linear1_linear = nn.Linear(d_model, class_num)
        self.linear2_cov = nn.Conv1d(d_model, 1, kernel_size=1)
        self.linear2_linear = nn.Linear(d_feature, class_num)
        self.bn=nn.BatchNorm1d(d_feature)
        self.bn2=nn.BatchNorm1d(16) 
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, src_seq):
        b, l = src_seq.size()
     
        src_pos = torch.LongTensor(
            [list(range(1, l + 1)) for i in range(b)]
        )
        src_pos = src_pos.to(self.device)

        enc_output, *_ = self.encoder(src_seq, src_pos)
        #print('encoder: ', enc_output.shape)
        
        res_llm = self.linear1_cov(enc_output)
        #print('linear1_cov: ', res.shape)

        res = res_llm.contiguous().view(res_llm.size()[0] , -1)
        #print('contig', res.shape)

        res = self.linear1_linear(res)
        #print('linear1_linear', res.shape)

        soft = self.softmax(res)

        return res, res_llm, soft


class BiLSTM(nn.Module):
    #vocab_size = 48, 32, 838
    def __init__(self, vocab_size, device, embedding_dim = SIG_LEN, hidden_dim1 = 128, hidden_dim2 = 768, output_dim = class_num, n_layers =2,
                 dropout = 0.3, bidirectional = True, pad_index = 1):
        super().__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx = pad_index)
        
        self.lstm = nn.LSTM(embedding_dim,
                            hidden_dim1,
                            num_layers=n_layers,
                            batch_first=True)
        self.fc1 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc2 = nn.Linear(hidden_dim2, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim = 1)

    def forward(self, text):

        packed_output, (hidden, cell) = self.lstm(text)

        rel = self.relu(packed_output)
        dense1 = self.fc1(rel)
        drop = self.dropout(dense1)
        preds = self.fc2(drop)
        soft = preds[:, -1, :]
        soft = self.softmax(soft)
        return preds, drop, soft

class MLP(nn.Module):
    def __init__(self, vocab_size, embed_size = 1, hidden_size2 = 256, hidden_size3 = 128, hidden_size4 = 768, 
    output_dim = class_num, dropout = 0.3, max_document_length = 32):
        super().__init__()

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(vocab_size, hidden_size2)  
        self.fc2 = nn.Linear(hidden_size2, hidden_size3) 
        self.fc3 = nn.Linear(hidden_size3, hidden_size4)  
        self.fc4 = nn.Linear(hidden_size4, output_dim)
        self.softmax = nn.Softmax(dim = 1)  

    def forward(self, text):
 
        x = self.relu(self.fc1(text))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.relu(self.fc3(x))
        x = self.dropout(x)
        preds = self.fc4(x)
        soft = self.softmax(preds)
        return preds, x, soft
        

class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    https://github.com/hsd1503/resnet1d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        net = self.conv(net)

        return net
        
class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    https://github.com/hsd1503/resnet1d
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class BasicBlock(nn.Module):
    """
    ResNet Basic Block
    https://github.com/hsd1503/resnet1d
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups, downsample, use_bn, use_do, is_first_block=False):
        super(BasicBlock, self).__init__()
        
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.stride = stride
        self.groups = groups
        self.downsample = downsample
        if self.downsample:
            self.stride = stride
        else:
            self.stride = 1
        self.is_first_block = is_first_block
        self.use_bn = use_bn
        self.use_do = use_do

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=self.stride,
            groups=self.groups)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)

    def forward(self, x):
        
        identity = x
        
        out = x
        if not self.is_first_block:
            if self.use_bn:
                out = self.bn1(out)
            out = self.relu1(out)
            if self.use_do:
                out = self.do1(out)
        out = self.conv1(out)
        
        if self.use_bn:
            out = self.bn2(out)
        out = self.relu2(out)
        if self.use_do:
            out = self.do2(out)
        out = self.conv2(out)
        
        if self.downsample:
            identity = self.max_pool(identity)
            
        if self.out_channels != self.in_channels:
            identity = identity.transpose(-1,-2)
            ch1 = (self.out_channels-self.in_channels)//2
            ch2 = self.out_channels-self.in_channels-ch1
            identity = F.pad(identity, (ch1, ch2), "constant", 0)
            identity = identity.transpose(-1,-2)
        
        out += identity

        return out
    
class ResNet1D(nn.Module):
    '''
    https://github.com/hsd1503/resnet1d
    '''
    def __init__(self, in_channels, base_filters, kernel_size, stride, groups, n_block, n_classes, downsample_gap=12, increasefilter_gap=12, use_bn=True, use_do=True, verbose=False):
        super(ResNet1D, self).__init__()
        
        self.verbose = verbose
        self.n_block = n_block
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.use_bn = use_bn
        self.use_do = use_do

        self.downsample_gap = downsample_gap 
        self.increasefilter_gap = increasefilter_gap 

        self.first_block_conv = MyConv1dPadSame(in_channels=in_channels, out_channels=base_filters, kernel_size=self.kernel_size, stride=1)
        self.first_block_bn = nn.BatchNorm1d(base_filters)
        self.first_block_relu = nn.ReLU()
        out_channels = base_filters
        self.softmax = nn.Softmax(dim = 1)
                
        self.basicblock_list = nn.ModuleList()
        for i_block in range(self.n_block):
            if i_block == 0:
                is_first_block = True
            else:
                is_first_block = False
       
            if i_block % self.downsample_gap == 1:
                downsample = True
            else:
                downsample = False
            if is_first_block:
                in_channels = base_filters
                out_channels = in_channels
            else:

                in_channels = int(base_filters*2**((i_block-1)//self.increasefilter_gap))
                if (i_block % self.increasefilter_gap == 0) and (i_block != 0):
                    out_channels = in_channels * 2
                else:
                    out_channels = in_channels
            
            tmp_block = BasicBlock(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=self.kernel_size, 
                stride = self.stride, 
                groups = self.groups, 
                downsample=downsample, 
                use_bn = self.use_bn, 
                use_do = self.use_do, 
                is_first_block=is_first_block)
            self.basicblock_list.append(tmp_block)

        self.final_bn = nn.BatchNorm1d(out_channels)
        self.final_relu = nn.ReLU(inplace=True)

        self.dense = nn.Linear(768, n_classes)

        
    def forward(self, x):
        
        out = x # (16, 1, 600)

        if self.verbose:
            print('input shape', out.shape)
        out = self.first_block_conv(out)    # (16,768,600)
        if self.verbose:
            print('after first conv', out.shape)
        if self.use_bn:
            out = self.first_block_bn(out)  # (16,768,600)
        out = self.first_block_relu(out)    # (16,768,600)
        
        for i_block in range(self.n_block):
            net = self.basicblock_list[i_block]
            if self.verbose:
                print('i_block: {0}, in_channels: {1}, out_channels: {2}, downsample: {3}'.format(i_block, net.in_channels, net.out_channels, net.downsample))
            out = net(out)  # (16,768,600)
            if self.verbose:
                print(out.shape)

        if self.use_bn:
            out = self.final_bn(out)    # (16,768,300)
        out = self.final_relu(out)  # (16,768,300)
        out_llm = out.mean(-1)  #  (16,768)
        # 因为我只需要这个模型编码，不需要做分类，所以下面的部分隐去
        # if self.verbose:
        #     print('final pooling', out_llm.shape)
        #
        # out = self.dense(out_llm)   #  (16,5)
        # if self.verbose:
        #     print('dense', out.shape)
        # soft = self.softmax(out)    #  (16,5)
        #
        # if self.verbose:
        #     print('softmax', out.shape)
        # out_pool = torch.mean(out_llm, dim=1)   #  (16,)
        return out, out_llm   # 最终输出， 编码特征， 最终输出的softmax版本, 编码特征池化输出

