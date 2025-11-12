import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
import torch.nn.functional as F
from models.fsgr.group_vit import gumbel_softmax, hard_softmax
from models.fsgr.group_vit import GroupingBlock
# from models.fsgr.encoders import VisualPromptEncoder
# from models.fsgr.attention import ScaledDotProductAttention
class MaskClipHead(nn.Module):
    def __init__(self, text_categories, text_channels, text_embeddings_path, gumbel_tau=1., assign_eps=1., ks_thresh=0., pd_thresh=0., training=False, gumbel=True, hard=True, sum_assign=False):
        super(MaskClipHead, self).__init__()
        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.training = training
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.gumbel_tau = gumbel_tau
        self.assign_eps = assign_eps

        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_channels))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_channels))
            self.text_embeddings.requires_grad = False
            self.load_text_embeddings()
        ### 此处是进行文本聚类的操作，将4764类文本特征聚类成为200个聚类中心
        # self.group_layer = GroupingBlock(512, 512, 8, num_group_token=200, norm_layer=nn.LayerNorm)
        # self.text_embeddings = torch.load(self.text_embeddings_path, map_location='cpu').unsqueeze(0)
        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cpu')
        self.text_embeddings[:, :] = loaded[:, :]
        print(f'Loaded text embeddings from {self.text_embeddings_path}')
    
    def forward(self, g_feat, input, tau, return_attn=False, return_index=False, visualize=False):
        ### 使用完整的库进行patch-level的检索
        # inputs.size (b_s, num_patch, dim)
        # [b_s, num_patch, 512] -> [b_s, 512, 14, 14]
        # feat = input.permute(0, 2, 1).contiguous().view(input.size(0), -1, 14, 14)

        ### 使用visual global feature预选出topk个相似的文本特征
        # [b_s, 1, 512] -> [b_s, 512, 1, 1]
        feat = input
        g_feat = g_feat.permute(0, 2, 1).contiguous().unsqueeze(-1)

        ### 此处是进行文本聚类的操作，将4764类文本特征聚类成为200个聚类中心
        # agg_text_features, text_attn_dict = self.group_layer(self.text_embeddings, pos=None, return_attn=True)
        ### 使用visual global feature预选出topk个相似的文本特征
        vg_logit = F.conv2d(g_feat, self.text_embeddings[:, :, None, None]).squeeze(-1).squeeze(-1)     # [b_s, num_text, 1, 1]
        indices = torch.topk(vg_logit, k=20, dim=1, largest=True)[1].unsqueeze(-1)   # [b_s, k]
        text_embeddings = self.text_embeddings.unsqueeze(0).expand(indices.size(0), -1, -1)
        agg_text_features = text_embeddings.gather(1, indices.expand(-1, -1, self.text_embeddings.size(-1)))    # (b_s, k, 512)
        seg_logit = torch.matmul(agg_text_features, feat.permute(0, 2, 1))   # (b_s, k, 512)(b_s, 512, 196)

        ### 计算出patch与文本之间的对应权重矩阵
        # seg_logit = F.conv2d(feat, self.text_embeddings[:, :, None, None])     # [b_s, num_text, 14, 14]

        seg_logit = tau.exp() * seg_logit.view(seg_logit.size(0), seg_logit.size(1), -1)
        attn, local_index = self.get_attn(seg_logit)     # [b_s, num_text, 196], [b_s, 196]
        # 将local_index映射回原始text_embeddings的索引
        index = indices.squeeze(-1).gather(1, local_index)  # (b_s, 196)
        if not self.sum_assign:
            assign_attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
            # print('after soft', torch.argmax(attn, dim=1))

        ### 按照权重矩阵将patch特征转变为网格化的text feats，注意有没有用global refine
        # (B, num_text, num_patch) -> (B, num_patch, num_text) @ (num_text, dim) = (B, num_patch, dim)
        # text_grids = assign_attn.transpose(1, 2) @ self.text_embeddings
        text_grids = assign_attn.transpose(1, 2) @ agg_text_features

        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(seg_logit, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
            return text_grids, attn_dict
        else:
            attn_dict = None
            if return_index or visualize:
                return text_grids, index
            else:
                return text_grids
        # if not self.training:
        #     output = self.refine_output(output, k)
    
    def clip_feature_surgery(self, image_features, g_feat, seg_logit, text_features, redundant_feats=None):
        # (b, nt, ni)
        if redundant_feats != None:
            redundant_logit = image_features @ redundant_feats.t()    # (b, ni, 1)
            redundant_logit = redundant_logit.permute(0, 2, 1)    # (b, 1, ni)
            seg_logit = seg_logit - redundant_logit
        else:
            # weights to restrain influence of obvious classes on others
            prob = g_feat @ text_features.t()
            prob = (prob * 2).softmax(-1)
            w = prob / prob.mean(-1, keepdim=True)

            # element-wise multiplied features
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
            seg_logit = seg_logit * w.reshape(b, n_t, 1)
            redundant_logit = seg_logit.mean(1, keepdim=True) # along cls dim
            seg_logit = seg_logit - redundant_logit

        return seg_logit
    
    # attn (B, num_text, num_patch)
    def get_attn(self, attn, gumbel=None, hard=None):
        if gumbel is None:
            gumbel = self.gumbel

        if hard is None:
            hard = self.hard

        attn_dim = -2
        if gumbel and self.training:
            # print("gumbel")
            attn = gumbel_softmax(attn, dim=attn_dim, hard=hard, tau=self.gumbel_tau)
            
        else:
            if hard:
                # print("hard")
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                # print(torch.argmax(attn, dim=1).size())
                # print('before soft', torch.argmax(attn, dim=1))
                attn = F.softmax(attn, dim=attn_dim)
                # print('after soft', torch.argmax(attn, dim=1))
        return attn

    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)
        return output

class MaskClipHead_org(nn.Module):

    def __init__(self, text_categories, text_channels, text_embeddings_path, gumbel_tau=1., assign_eps=1., ks_thresh=0., pd_thresh=0., training=False, gumbel=True, hard=True, sum_assign=False):
        super(MaskClipHead_org, self).__init__()
        self.text_categories = text_categories
        self.text_channels = text_channels
        self.text_embeddings_path = text_embeddings_path
        self.training = training
        self.hard = hard
        self.gumbel = gumbel
        self.sum_assign = sum_assign
        self.gumbel_tau = gumbel_tau
        self.assign_eps = assign_eps

        if self.text_embeddings_path is None:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_channels))
            nn.init.normal_(self.text_embeddings, mean=0.0, std=0.01)
        else:
            self.text_embeddings = nn.Parameter(torch.zeros(text_categories, text_channels))
            self.text_embeddings.requires_grad = False
            self.load_text_embeddings()
        ### 此处是进行文本聚类的操作，将4764类文本特征聚类成为200个聚类中心
        # self.group_layer = GroupingBlock(512, 512, 8, num_group_token=200, norm_layer=nn.LayerNorm)
        # self.text_embeddings = torch.load(self.text_embeddings_path, map_location='cpu').unsqueeze(0)
        self.ks_thresh = ks_thresh
        self.pd_thresh = pd_thresh

    def load_text_embeddings(self):
        loaded = torch.load(self.text_embeddings_path, map_location='cpu')
        self.text_embeddings[:, :] = loaded[:, :]
        print(f'Loaded text embeddings from {self.text_embeddings_path}')
    
    def forward(self, input, g_feat=None, return_attn=False):
        # inputs.size (b_s, num_patch, dim)
        # [b_s, num_patch, 512] -> [b_s, 512, 14, 14]
        feat = input.permute(0, 2, 1).contiguous().view(input.size(0), -1, 14, 14)
        feat = feat / feat.norm(dim=1, keepdim=True)
        input = input / input.norm(dim=-1, keepdim=True)

        ### 此处是进行文本聚类的操作，将4764类文本特征聚类成为200个聚类中心
        # agg_text_features, text_attn_dict = self.group_layer(self.text_embeddings, pos=None, return_attn=True)
        # agg_text_features = agg_text_features.squeeze(0)
        
        ### 计算出patch与文本之间的对应权重矩阵
        seg_logit = F.conv2d(feat, self.text_embeddings[:, :, None, None])     # [b_s, num_text, 14, 14]
        seg_logit = seg_logit.view(seg_logit.size(0), seg_logit.size(1), -1)
        if g_feat is not None:
            seg_logit = self.clip_feature_surgery(input, g_feat, seg_logit, self.text_embeddings)
        attn = self.get_attn(seg_logit)     # [b_s, num_text, 196]
        ### 按照权重矩阵将patch特征转变为网格化的text feats
        if not self.sum_assign:
            assign_attn = attn / (attn.sum(dim=-1, keepdim=True) + self.assign_eps)
            # print('after soft', torch.argmax(attn, dim=1))
        # (B, num_text, num_patch) -> (B, num_patch, num_text) @ (num_text, dim) = (B, num_patch, dim)
        text_grids = assign_attn.transpose(1, 2) @ self.text_embeddings
        if return_attn:
            hard_attn = attn.clone()
            soft_attn = self.get_attn(seg_logit, gumbel=False, hard=False)
            attn_dict = {'hard': hard_attn, 'soft': soft_attn}
            return text_grids, attn_dict
        else:
            attn_dict = None
            return text_grids
        # if not self.training:
        #     output = self.refine_output(output, k)
    
    def clip_feature_surgery(self, image_features, g_feat, seg_logit, text_features, redundant_feats=None):
        # (b, nt, ni)
        if redundant_feats != None:
            redundant_logit = image_features @ redundant_feats.t()    # (b, ni, 1)
            redundant_logit = redundant_logit.permute(0, 2, 1)    # (b, 1, ni)
            seg_logit = seg_logit - redundant_logit
            
        else:
            # weights to restrain influence of obvious classes on others
            prob = g_feat @ text_features.t()
            prob = (prob * 2).softmax(-1)
            w = prob / prob.mean(-1, keepdim=True)

            # element-wise multiplied features
            b, n_t, n_i, c = image_features.shape[0], text_features.shape[0], image_features.shape[1], image_features.shape[2]
            seg_logit = seg_logit * w.reshape(b, n_t, 1)
            redundant_logit = seg_logit.mean(1, keepdim=True) # along cls dim
            seg_logit = seg_logit - redundant_logit

        return seg_logit
    
    # attn (B, num_text, num_patch)
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
                # print(torch.argmax(attn, dim=1).size())
                # print('before soft', torch.argmax(attn, dim=1))
                attn = F.softmax(attn, dim=attn_dim)
                # print('after soft', torch.argmax(attn, dim=1))
        return attn

    def refine_output(self, output, k):
        if self.pd_thresh > 0:
            N, C, H, W = output.shape
            _output = F.softmax(output*100, dim=1)
            max_cls_conf = _output.view(N, C, -1).max(dim=-1)[0]
            selected_cls = (max_cls_conf < self.pd_thresh)[:, :, None, None].expand(N, C, H, W)
            output[selected_cls] = -100

        if k is not None and self.ks_thresh > 0:
            output = F.softmax(output*100, dim=1)
            N, C, H, W = output.shape
            output = output.view(N, C, -1).transpose(-2, -1)
            # softmax
            # weight = k @ k.transpose(-2, -1)
            # weight = F.softmax(weight, dim=-1)
            # L2 distance
            k = F.normalize(k, p=2)
            weight = k @ k.transpose(-2, -1)

            selected_pos = (output.max(dim=-1, keepdim=True)[0] < self.ks_thresh)
            selected_pos = selected_pos.expand(-1, -1, C)

            weighted_output = weight @ output
            output[selected_pos] = weighted_output[selected_pos]
            output = output.transpose(-2, -1).view(N, C, H, W)

        return output