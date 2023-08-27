import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

try:
    import torch.distributed.nn
    from torch import distributed as dist
    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None

class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=False,
            rank=0,
            world_size=1,
            use_horovod=False,
            temperature=0.07
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temperature = temperature
        # cache state
        self.prev_num_logits = 0
        self.labels = {}
        
    def forward(self, image_features, text_features):
        logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        device = image_features.device
        logits_per_image = logit_scale * image_features @ text_features.T   # (32,32)
        logits_per_text = logit_scale * text_features @ image_features.T    # (32,32)

        # calculated ground-truth and cache if enabled
        num_logits = logits_per_image.shape[0]
        labels = torch.eye(num_logits, device=device, dtype=torch.float)    # (32,32)
        pred_1 = F.log_softmax(logits_per_image,dim=-1) # (32, 32)
        pred_2 = F.log_softmax(logits_per_text,dim=-1)
        loss_a = F.kl_div(pred_1, labels,reduction = 'sum')/num_logits  # ()
        loss_b = F.kl_div(pred_2, labels,reduction = 'sum')/num_logits
        total_loss = (loss_a + loss_b)/2
        return total_loss

class UniCL(nn.Module):
    def __init__(self,
                 local_loss=False,
                gather_with_grad=False,
                cache_labels=False,
                rank=0,
                world_size=1,
                use_horovod=False,
                temperature=0.05,
                uniCl_type="increase_dimension"):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod
        self.temperature = temperature
        self.uniCl_type = uniCl_type
        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def forward(self, image_features, text_features, labels): # (32,768), (32,768), (32,5)
        device = image_features.device
        batch_size, label_nums = labels.shape

        y = torch.ones((labels.shape[1], labels.shape[0], labels.shape[0]), device=device)    # (5, 32, 32)
        for i in range(labels.shape[0]):
            for j in range(labels.shape[1]):
                if labels[i, j] == 0:
                    y[j, i, :] = 0
                    y[j, :, i] = 0
        logits_per_image = image_features @ text_features.T  # logits_per_image维度为：(32,32)
        logits_per_text = text_features @ image_features.T  # (32,32)
        num_logits = logits_per_image.shape[1]

        if self.uniCl_type == "increase_dimension": # AugCL
            logit_scale = nn.Parameter(F.normalize(torch.ones([label_nums]), p=2, dim=0) * np.log(1 / self.temperature))
            logits_per_image = logits_per_image.unsqueeze(0).repeat(label_nums,1,1)      # (5,32,32)
            logits_per_text = logits_per_text.unsqueeze(0).repeat(label_nums, 1, 1)      # (5,32,32)
            for i in range(label_nums):
                logits_per_image[i] = logit_scale[i] * logits_per_image[i].clone()
                logits_per_text[i] = logit_scale[i] * logits_per_text[i].clone()

            total_loss = torch.tensor(0.0, device=device)
            for image, text, label in zip(logits_per_image, logits_per_text, y):
                pred_1 = F.log_softmax(image, dim=-1)
                pred_2 = F.log_softmax(text, dim=-1)
                loss_a = F.kl_div(pred_1, label, reduction='sum') / num_logits
                loss_b = F.kl_div(pred_2, label, reduction='sum') / num_logits
                loss = (loss_a + loss_b) / 2
                total_loss = torch.add(total_loss, loss)

        else:   # UniCL
            logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
            logits_per_image = logit_scale * logits_per_image
            logits_per_text = logit_scale * logits_per_text
            uni_labels = torch.max(y, dim=0).values
            pred_1 = F.log_softmax(logits_per_image, dim=-1)  # (32, 32)
            pred_2 = F.log_softmax(logits_per_text, dim=-1)
            loss_a = F.kl_div(pred_1, uni_labels, reduction='sum') / num_logits  # ()
            loss_b = F.kl_div(pred_2, uni_labels, reduction='sum') / num_logits
            total_loss = (loss_a + loss_b) / 2
        return  total_loss / label_nums

if __name__ == '__main__':
    # logit_scale = nn.Parameter(torch.ones([3]) * np.log(1 / 0.07))
    # matrix = torch.randn((3,2,2))
    # print(logit_scale * matrix)
    test = UniCL()
    result = test(torch.randn((3,5)), torch.randn((3,5)), torch.randint(0,2,(3,4)))


