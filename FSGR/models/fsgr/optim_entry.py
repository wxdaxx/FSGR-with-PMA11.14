import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F


def build_optimizer(model):
    """
    分两组：
    - backbone 组：冻结大部分参数，只放开 S_Adapter / prompt，lr 较小
    - decoders/encoders 组：其余参数，lr 较大
    注意：本函数仅设置“初始 base lr”，训练脚本会据 args.xe_base_lr 再覆盖。
    """
    params_enc, params_dec = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'backbone' in name:
            params_enc.append(param)  # backbone 组
        else:
            params_dec.append(param)  # decoder/encoder 组

    all_params = [
        {'params': params_enc, 'lr': 0.1},  # 会被外部覆盖成 args.xe_base_lr * 0.1
        {'params': params_dec, 'lr': 1.0},  # 会被外部覆盖成 args.xe_base_lr
    ]
    optimizer = Adam(all_params, lr=1.0, betas=(0.9, 0.98))
    return optimizer


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss (arXiv:2004.11362)"""
    def __init__(self, temperature=0.07, contrast_mode='all', base_temperature=0.07):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')

        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature)
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        mask = mask.repeat(anchor_count, contrast_count)
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6,
                                     torch.ones_like(mask_pos_pairs),
                                     mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss

    def forward_similarity(self, features, labels=None, mask=None):
        device = torch.device('cuda') if features.is_cuda else torch.device('cpu')
        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...]')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)

        contrast_count = features.shape[1]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor_feature = contrast_feature
        anchor_count = contrast_count

        normalized_anchor = F.normalize(anchor_feature, p=2, dim=1)
        normalized_contrast = F.normalize(contrast_feature, p=2, dim=1)
        anchor_dot_contrast = torch.matmul(normalized_anchor, normalized_contrast.T)

        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        mask = mask * logits_mask

        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6,
                                     torch.ones_like(mask_pos_pairs),
                                     mask_pos_pairs)
        mean_log_prob_pos = (mask * anchor_dot_contrast).sum(1) / mask_pos_pairs

        loss = mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()
        return loss
