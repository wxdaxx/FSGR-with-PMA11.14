import os
import math
from functools import reduce
from operator import mul

import torch
from torch import nn

from models.containers import ModuleList
from ..captioning_model import CaptioningModel
from models.fsgr.grid_aug import PositionEmbeddingSine
from models.fsgr.sclip_model_dpt import build_model
from models.fsgr.projection import MaskClipHead
from timm.models.layers import trunc_normal_


class Transformer(CaptioningModel):
    def __init__(self, bos_idx, encoder, decoder,
                 adapter_layer_list, pre_vs_path, text_emb_path,
                 pre_name, text, return_index=False, archi_edit=False, d_in=1280):
        super(Transformer, self).__init__()
        self.bos_idx = bos_idx
        self.encoder = encoder
        self.decoder = decoder
        self.pre_name = pre_name
        self.archi_edit = archi_edit
        self.pre_vs_path = pre_vs_path
        self.grid_embedding = PositionEmbeddingSine(self.decoder.d_model // 2, normalize=True)

        self.register_state('enc_output', None)
        self.register_state('mask_enc', None)
        self.init_weights()
        self.text = text
        self.return_index = return_index

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if self.text:
            self.att_embed = nn.Sequential(
                nn.Linear(d_in, self.d_model),
                nn.CELU(1.3, inplace=True),
                nn.Dropout(0.1),
                nn.LayerNorm(self.d_model)
            )
            # 这里 text_categories=80（COCO 80类）
            self.text_embed = MaskClipHead(
                text_categories=80, text_channels=512,
                text_embeddings_path=text_emb_path, training=True
            )

        # 兼容加载 .pt（TorchScript 或 state_dict）
        try:
            model = torch.jit.load(self.pre_vs_path, map_location='cpu').float()
        except (RuntimeError, ValueError):
            model = torch.load(self.pre_vs_path, map_location='cpu').float()

        if self.archi_edit:
            self.backbone = build_model('CS-' + pre_name, model.state_dict(), adapter_layer_list).to(device).float()
        else:
            self.backbone = build_model(pre_name, model.state_dict(), adapter_layer_list).to(device).float()

        torch.cuda.empty_cache()

        val = math.sqrt(6. / float(3 * reduce(mul, [16, 16], 1) + 768))

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        print('load pretrained weights!')

        # 冻结除 S_Adapter / prompt 以外参数
        for n, p in self.backbone.named_parameters():
            if 'S_Adapter' in n or 'prompt' in n:
                p.requires_grad = True
            else:
                p.requires_grad = False

        nn.init.uniform_(self.backbone.visual.deep_prompt_embeddings.data, -val, val)
        nn.init.uniform_(self.backbone.visual.prompt_embeddings.data, -val, val)

        for n, mod in self.backbone.visual.named_modules():
            if 'S_Adapter' in n:
                for n2, m2 in mod.named_modules():
                    if 'D_fc2' in n2 and isinstance(m2, nn.Linear):
                        nn.init.constant_(m2.weight, 0)
                        nn.init.constant_(m2.bias, 0)
                    else:
                        _init_weights(m2)
            elif 'prompt_proj' in n:
                for _, m in mod.named_modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, a=0, mode='fan_out')

    @property
    def d_model(self):
        return self.decoder.d_model

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def get_pos_embedding(self, grids):
        bs = grids.shape[0]
        grid_embed = self.grid_embedding(grids.view(bs, 14, 14, -1))
        return grid_embed

    def forward(self, images, seq, *args):
        """
        images: (B, 3, 224, 224)
        seq:    (B, L)
        """
        att_feats, global_feat, align_feats = self.backbone.encode_image(images, return_all=True, csa=True)
        att_feats = att_feats / att_feats.norm(dim=-1, keepdim=True)
        global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
        align_feats = align_feats / align_feats.norm(dim=-1, keepdim=True)

        global_feat = global_feat.unsqueeze(1)

        if self.text:
            g_feat = global_feat.expand(-1, att_feats.size(1), -1)
            att_feats_cat = torch.cat([att_feats, g_feat], dim=-1)
            att_feats_cat = self.att_embed(att_feats_cat)
            result = self.text_embed(global_feat, align_feats, self.backbone.logit_scale, return_index=self.return_index)
            if isinstance(result, tuple):
                text_grids, index = result
                index = index.reshape(index.size(0) * index.size(1), -1)
            else:
                text_grids = result
            contrast_feats = att_feats_cat / att_feats_cat.norm(dim=-1, keepdim=True)
            contrast_feats = contrast_feats.reshape(att_feats_cat.size(0) * att_feats_cat.size(1), -1).unsqueeze(1)
        else:
            text_grids = att_feats

        grid_embed = self.get_pos_embedding(att_feats)
        enc_output, mask_enc = self.encoder(att_feats, global_feat, text_grids, pos=grid_embed)

        # 调试开关
        if os.environ.get("FSGR_DEBUG") == "1":
            print(f"[DEBUG] seq.shape={seq.shape}, enc_output={enc_output.shape}")

        dec_output = self.decoder(0, seq, enc_output, mask_enc, pos=grid_embed)

        if self.return_index:
            return dec_output, contrast_feats, index
        else:
            return dec_output

    def init_state(self, b_s, device):
        return [torch.zeros((b_s, 0), dtype=torch.long, device=device), None, None]

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        it = None
        if mode == 'teacher_forcing':
            raise NotImplementedError
        elif mode == 'feedback':
            if t == 0:
                with torch.no_grad():
                    att_feats, global_feat, align_feats = self.backbone.encode_image(visual, return_all=True, csa=True)
                att_feats = att_feats / att_feats.norm(dim=-1, keepdim=True)
                global_feat = global_feat / global_feat.norm(dim=-1, keepdim=True)
                align_feats = align_feats / align_feats.norm(dim=-1, keepdim=True)
                global_feat = global_feat.unsqueeze(1)

                if self.text:
                    g_feat = global_feat.expand(-1, att_feats.size(1), -1)
                    att_feats_cat = torch.cat([att_feats, g_feat], dim=-1)
                    att_feats_cat = self.att_embed(att_feats_cat)
                    result = self.text_embed(global_feat, align_feats, self.backbone.logit_scale)
                    if isinstance(result, tuple):
                        text_grids, index = result
                        index = index.reshape(index.size(0) * index.size(1), -1)
                    else:
                        text_grids = result
                else:
                    text_grids = att_feats

                self.grid_embed = self.get_pos_embedding(att_feats)
                self.enc_output, self.mask_enc = self.encoder(att_feats, global_feat, text_grids, pos=self.grid_embed)

                if isinstance(visual, torch.Tensor):
                    it = visual.data.new_full((visual.shape[0], 1), self.bos_idx).long()
                else:
                    it = visual[0].data.new_full((visual[0].shape[0], 1), self.bos_idx).long()
            else:
                it = prev_output

        return self.decoder(t, it, self.enc_output, self.mask_enc, pos=self.grid_embed)


class TransformerEnsemble(CaptioningModel):
    def __init__(self, model: Transformer, weight_files):
        super(TransformerEnsemble, self).__init__()
        self.n = len(weight_files)
        self.models = ModuleList([torch.nn.utils.parametrize.deepcopy(model) for _ in range(self.n)])
        for i in range(self.n):
            state_dict_i = torch.load(weight_files[i])['state_dict']
            self.models[i].load_state_dict(state_dict_i)

    def step(self, t, prev_output, visual, seq, mode='teacher_forcing', **kwargs):
        out_ensemble = []
        for i in range(self.n):
            out_i = self.models[i].step(t, prev_output, visual, seq, mode, **kwargs)
            out_ensemble.append(out_i.unsqueeze(0))
        return torch.mean(torch.cat(out_ensemble, 0), dim=0)
