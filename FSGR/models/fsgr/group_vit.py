# -------------------------------------------------------------------------
# Copyright (c) 2021-2022, NVIDIA Corporation & Affiliates. All rights reserved.
#
# This work is made available under the Nvidia Source Code License.
# To view a copy of this license, visit
# https://github.com/NVlabs/GroupViT/blob/main/LICENSE
#
# Written by Jiarui Xu
# -------------------------------------------------------------------------

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from models.fsgr.encoders import EncoderLayer

class Mlp(nn.Module):

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret, index.squeeze(-2)


def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (-torch.empty_like(
    #     logits,
    #     memory_format=torch.legacy_contiguous_format).exponential_().log()
    #             )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)
    origin_index = logits.max(dim, keepdim=True)[1]

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, origin_index.squeeze(-2)

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 out_dim=None,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        else:
            B, N, C = query.shape
            if key is None:
                key = query
            if value is None:
                value = key
            S = key.size(1)
            # [B, nh, N, C//nh]
            q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
            # [B, nh, S, C//nh]
            v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out


class CrossAttnBlock(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 post_norm=False):
        super().__init__()
        if post_norm:
            self.norm_post = norm_layer(dim)
            self.norm_q = nn.Identity()
            self.norm_k = nn.Identity()
        else:
            self.norm_q = norm_layer(dim)
            self.norm_k = norm_layer(dim)
            self.norm_post = nn.Identity()
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key, *, mask=None):
        x = query
        x = x + self.drop_path(self.attn(self.norm_q(query), self.norm_k(key), mask=mask))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        x = self.norm_post(x)
        return x


class AssignAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=1,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 hard=True,
                 gumbel=False,
                 gumbel_tau=1.,
                 sum_assign=False,
                 assign_eps=1.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.gumbel = gumbel
        self.gumbel_tau = gumbel_tau
        self.sum_assign = sum_assign
        self.assign_eps = assign_eps

    def get_attn(self, attn, gumbel=None, hard=None):

        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
        else:
            if hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, key=None, *, value=None, return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c', h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c', h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        raw_attn = (q @ k.transpose(-2, -1)) * self.scale

        attn = self.get_attn(raw_attn)
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(raw_attn, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
        else:
            attn_dict = None

        if not self.sum_assign:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B, n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out, attn_dict


class GroupingBlock(nn.Module):
    """Grouping Block to group similar segments together.

    Args:
        dim (int): Dimension of the input.
        out_dim (int): Dimension of the output.
        num_heads (int): Number of heads in the grouping attention.
        num_output_group (int): Number of output groups.
        norm_layer (nn.Module): Normalization layer to use.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        hard (bool): Whether to use hard or soft assignment. Default: True
        gumbel (bool): Whether to use gumbel softmax. Default: True
        sum_assign (bool): Whether to sum assignment or average. Default: False
        assign_eps (float): Epsilon to avoid divide by zero. Default: 1
        gum_tau (float): Temperature for gumbel softmax. Default: 1
    """

    def __init__(self,
                 dim,
                 out_dim,
                 num_heads,
                 num_group_token,
                 norm_layer,
                 mlp_ratio=(0.5, 4.0),
                 hard=True,
                 gumbel=True,
                 sum_assign=False,
                 assign_eps=1.,
                 gumbel_tau=1.,
                 zero_init_group_token=False):
        super(GroupingBlock, self).__init__()
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        # norm on group_tokens
        self.norm_tokens = norm_layer(dim)
        _, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.num_group_token = num_group_token
        if num_group_token > 0:
            self.group_token = nn.Parameter(torch.zeros(1, num_group_token, dim))
            if not zero_init_group_token:
                trunc_normal_(self.group_token, std=.02)
        else:
            self.group_token = None
        # norm on x
        self.norm_x = norm_layer(dim)
        d_k = dim // num_heads
        d_v = d_k
        self.pre_assign_attn = EncoderLayer(dim, d_k, d_v, num_heads, d_ff=2048, dropout=.1)
        # self.pre_assign_attn = CrossAttnBlock(
        #     dim=dim, num_heads=num_heads, mlp_ratio=4, qkv_bias=True, norm_layer=norm_layer, post_norm=True)

        self.assign = AssignAttention(
            dim=dim,
            num_heads=1,
            qkv_bias=True,
            hard=hard,
            gumbel=gumbel,
            gumbel_tau=gumbel_tau,
            sum_assign=sum_assign,
            assign_eps=assign_eps)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim), nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def forward(self, x, pos, return_attn=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            group_tokens (torch.Tensor): group tokens, [B, S_1, C]
            return_attn (bool): whether to return attention map

        Returns:
            new_x (torch.Tensor): [B, S_1, C], S_1 is the number of group tokens
        """
        group_tokens = self.group_token.expand(x.size(0), -1, -1)
        group_tokens = self.norm_tokens(group_tokens)
        x = self.norm_x(x)
        # [B, S_1, C]
        
        projected_group_tokens = self.pre_assign_attn(group_tokens, x, x, pos=pos)
        new_x, attn_dict = self.assign(projected_group_tokens, x, return_attn=return_attn)
        new_x += projected_group_tokens

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, attn_dict