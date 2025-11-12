from torch.nn import functional as F
from models.fsgr.utils import PositionWiseFeedForward
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
from models.fsgr.attention import MultiHeadGeometryAttention
from models.fsgr.grid_aug import BoxRelationalEmbedding

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, pre_sw_path=None, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                                attention_module=attention_module,
                                                attention_module_kwargs=attention_module_kwargs)
        # self.mhatt2 = MultiHeadGeometryAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
        #                                         attention_module=attention_module,
        #                                         attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, relative_geometry_weights=None, attention_mask=None, attention_weights=None, pos=None):

        # q, k = (queries + pos, keys + pos) if pos is not None else (queries, keys)
        
        if pos is not None:
            if queries.size(1) != keys.size(1):
                # q = queries[:, 1:, :] + pos
                # q = torch.cat((queries[:, :1, :], q), dim=1)
                q = queries + pos
            else:
                # q = queries[:, 1:, :] + pos
                # q = torch.cat((queries[:, :1, :], q), dim=1)
                q = queries + pos
            # k = keys[:, 1:, :] + pos
            # k = torch.cat((keys[:, :1, :], k), dim=1)
            # q = queries + pos
            k = keys + pos
        else:
            q = queries
            k = keys
        att = self.mhatt(q, k, values, relative_geometry_weights, attention_mask, attention_weights, need_attn=False)
        # att2 = self.mhatt2(q, k, keys, relative_geometry_weights, attention_mask, attention_weights, need_attn=False)
        # att = att1 + att2
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.h = h 
        
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,
                                                  attention_module_kwargs=attention_module_kwargs)
                                     for _ in range(N)])
        self.padding_idx = padding_idx

        self.WGs = nn.ModuleList([nn.Linear(64, 1, bias=True) for _ in range(h)])

    def forward(self, input, pixels, attention_weights=None, pos=None, text=True):
        # input (b_s, seq_len, d_in)
        if self.padding_idx is not None:
            attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
            attention_mask_k = (torch.sum(pixels, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, seq_len)
        else:
            attention_mask = torch.zeros((input.size(0), input.size(1))).unsqueeze(1).unsqueeze(1).bool().to(device)
            attention_mask_k = torch.zeros((pixels.size(0), pixels.size(1))).unsqueeze(1).unsqueeze(1).bool().to(device)  # (b_s, 1, 1, seq_len)
        # grid geometry embedding
        # follow implementation of https://github.com/yahoo/object_relation_transformer/blob/ec4a29904035e4b3030a9447d14c323b4f321191/models/RelationTransformerModel.py
        if pos is not None:
            relative_geometry_embeddings = BoxRelationalEmbedding(pixels)
            flatten_relative_geometry_embeddings = relative_geometry_embeddings.view(-1, 64)
            box_size_per_head = list(relative_geometry_embeddings.shape[:3])
            box_size_per_head.insert(1, 1)
            relative_geometry_weights_per_head = [layer(flatten_relative_geometry_embeddings).view(box_size_per_head) for layer in self.WGs]
            relative_geometry_weights = torch.cat((relative_geometry_weights_per_head), 1)
            relative_geometry_weights = F.relu(relative_geometry_weights)
        else:
            relative_geometry_weights = None

        # relative_geometry_weights = None
        query = input
        k_v = pixels
        # print(query.size())
        # print(k_v.size())
        for layer in self.layers:
            # if query.equal(k_v):
            #     print('done')
            #     print(query.size())
            query = layer(query, k_v, k_v, relative_geometry_weights, attention_mask_k, attention_weights, pos=pos)
            if text:
                pass
            else:
                k_v = query
        return query, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, text, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, **kwargs)
        if text:
            self.pixel_embed = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.CELU(1.3, inplace=True),
                nn.Dropout(0.1),
                nn.LayerNorm(self.d_model)
            )
        else:
            self.pixel_embed = nn.Sequential(
                nn.Linear(768, self.d_model),
                nn.CELU(1.3, inplace=True),
                nn.Dropout(0.1),
                nn.LayerNorm(self.d_model)
            )
        self.global_embed = nn.Sequential(
                nn.Linear(self.d_model, self.d_model),
                nn.CELU(1.3, inplace=True),
                nn.Dropout(0.1),
                nn.LayerNorm(self.d_model)
            )
        self.layer_norm = nn.LayerNorm(self.d_model)
        self.text = text

    def forward(self, att_feats, g_feat, text_grids, attention_weights=None, pos=None):
        # # input--已经降维的图像网格特征，g_feat--可能会有的全局特征，text_grids--计算得到的text embedding网格
        # g_feat = self.global_embed(g_feat)
        # att_feats = torch.cat([g_feat, att_feats], dim=1)

        pixels = self.pixel_embed(text_grids)
        # pixels = torch.cat([g_feat, pixels], dim=1)
        pixels_mask = (torch.sum(pixels, dim=-1) == 0).unsqueeze(-1)
        pixels = self.layer_norm(pixels)
        pixels = pixels.masked_fill(pixels_mask, 0)
        if self.text:
            att_mask = (torch.sum(att_feats, dim=-1) == 0).unsqueeze(-1)
            att_feats = self.layer_norm(att_feats)
            att_feats_1 = att_feats.masked_fill(att_mask, 0)
        else:
            # 如果没使用text embeding，那原始的grid feats就还没经历降维，在这里需要做，又因为输入的text_grids=att_feats，所以可以直接用
            att_feats_1 = pixels
            # print('no text')
        return super(TransformerEncoder, self).forward(att_feats_1, pixels, attention_weights=attention_weights, pos=pos, text=self.text)

# class VisualPromptEncoder(MultiLevelEncoder):
#     def __init__(self, N, padding_idx=None, **kwargs):
#         super(VisualPromptEncoder, self).__init__(N, padding_idx=None, **kwargs)
#         self.pixel_embed = nn.Sequential(
#             nn.Linear(768, self.d_model),
#             nn.CELU(1.3, inplace=True),
#             nn.Dropout(0.1),
#             nn.LayerNorm(self.d_model)
#         )
#         self.layer_norm = nn.LayerNorm(self.d_model)

#     def forward(self, text_feats, g_feat, visual_prompts, attention_weights=None, pos=None):
#         # # text_feats--提取出来的concept文本特征，g_feat--可能会有的全局特征，visual_prompts--从CLIP得到的visual prompt
#         visual_prompts = self.pixel_embed(visual_prompts)
#         visual_prompts = torch.cat([g_feat, visual_prompts], dim=1)
#         visual_prompts = self.layer_norm(visual_prompts)
        
#         return super(VisualPromptEncoder, self).forward(text_feats, visual_prompts, attention_weights=attention_weights, pos=pos, text=False)
