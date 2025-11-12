import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from torch import nn
from torch.nn import functional as F

from models.fsgr.attention import MultiHeadAttention
from models.fsgr.utils import sinusoid_encoding_table, PositionWiseFeedForward
from models.fsgr.attention import MultiHeadAdaptiveAttention
from models.containers import Module, ModuleList


def _build_aligned_embedding(vocab_size, d_model, padding_idx, embed_path='word_embeds.pth', key='clip_embeds'):
    """
    尝试加载预训练嵌入；若维度或行数与 vocab 不一致，则对齐/填充/裁剪。
    - 对齐规则：new[vocab_size, d]，将预训练[:min, :] 拷入，多余行按N(0,0.02)随机初始化
    - <pad> 行置零
    - 若文件缺失或key不存在，则直接随机初始化
    """
    new_weight = None
    try:
        blob = torch.load(embed_path, map_location='cpu')
        if key not in blob:
            raise KeyError(f"'{key}' not in {embed_path}")
        pre = blob[key].float()  # [N_pre, d_pre]
        d_model_file = pre.shape[1]
        if d_model is not None and d_model != d_model_file:
            # 以文件中维度为准，后续层用d_model_file
            d = d_model_file
        else:
            d = d_model if d_model is not None else d_model_file

        new_weight = torch.empty(vocab_size, d).normal_(mean=0.0, std=0.02)
        rows = min(vocab_size, pre.shape[0])
        cols = min(d, pre.shape[1])
        new_weight[:rows, :cols] = pre[:rows, :cols]
        if padding_idx is not None and 0 <= padding_idx < vocab_size:
            new_weight[padding_idx].zero_()
    except Exception as e:
        # 回退：随机初始化
        d = d_model if d_model is not None else 512
        new_weight = torch.empty(vocab_size, d).normal_(mean=0.0, std=0.02)
        if padding_idx is not None and 0 <= padding_idx < vocab_size:
            new_weight[padding_idx].zero_()
    return new_weight


class DecoderLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)
        self.enc_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                          attention_module=enc_att_module,
                                          attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, t, input, enc_output, mask_pad, mask_self_att, mask_enc_att, pos):
        # 自注意力 + 残差层归一化
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad

        # 编码器-解码器注意力 + 残差层归一化
        k = enc_output + pos
        enc_att = self.enc_att(self_att, k, enc_output, mask_enc_att, t=t, need_attn=False)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad

        # 前馈 + 残差层归一化
        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class DecoderAdaptiveLayer(Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, self_att_module=None,
                 enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(DecoderAdaptiveLayer, self).__init__()
        self.self_att = MultiHeadAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=True,
                                           attention_module=self_att_module,
                                           attention_module_kwargs=self_att_module_kwargs)

        self.enc_att = MultiHeadAdaptiveAttention(d_model, d_k, d_v, h, dropout, can_be_stateful=False,
                                                  attention_module=enc_att_module,
                                                  attention_module_kwargs=enc_att_module_kwargs)

        self.dropout1 = nn.Dropout(dropout)
        self.lnorm1 = nn.LayerNorm(d_model)

        self.dropout2 = nn.Dropout(dropout)
        self.lnorm2 = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout)

    def forward(self, input, enc_output, mask_pad, mask_self_att, mask_enc_att, language_feature=None, pos=None, drop=1):
        self_att = self.self_att(input, input, input, mask_self_att)
        self_att = self.lnorm1(input + self.dropout1(self_att))
        self_att = self_att * mask_pad

        key = enc_output + pos
        enc_att = self.enc_att(self_att, key, enc_output, mask_enc_att,
                               language_feature=language_feature, drop=drop)
        enc_att = self.lnorm2(self_att + self.dropout2(enc_att))
        enc_att = enc_att * mask_pad

        ff = self.pwff(enc_att)
        ff = ff * mask_pad
        return ff


class TransformerDecoderLayer(Module):
    def __init__(self, vocab_size, max_len, N_dec, padding_idx,
                 language_model_path='/home/gaojc/project/RSTNet/saved_language_models/bert_language_best.pth',
                 d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, bert_hidden_size=768, dropout=.1,
                 self_att_module=None, enc_att_module=None, self_att_module_kwargs=None, enc_att_module_kwargs=None):
        super(TransformerDecoderLayer, self).__init__()
        self._is_stateful = False
        self.d_model = d_model
        self.padding_idx = padding_idx
        self.max_len = max_len
        self.N = N_dec

        # 词嵌入（对齐容错）
        weight = _build_aligned_embedding(vocab_size, d_model, padding_idx,
                                          embed_path='word_embeds.pth', key='clip_embeds')
        self.d_model = weight.shape[1]  # 以实际嵌入维度为准
        self.word_emb = nn.Embedding.from_pretrained(weight, freeze=False, padding_idx=padding_idx)

        # 位置嵌入
        self.pos_emb = nn.Embedding.from_pretrained(
            sinusoid_encoding_table(max_len + 1, self.d_model, 0), freeze=False
        )

        # 解码层：默认 N_dec+1 个标准层（保持与原始实现一致）
        self.layers = ModuleList([
            DecoderLayer(self.d_model, d_k, d_v, h, d_ff, dropout,
                         self_att_module=self_att_module, enc_att_module=enc_att_module,
                         self_att_module_kwargs=self_att_module_kwargs,
                         enc_att_module_kwargs=enc_att_module_kwargs)
            for _ in range(N_dec + 1)
        ])

        # 输出投影
        self.fc = nn.Linear(self.d_model, vocab_size, bias=False)

        # 状态缓存
        self.register_state('running_mask_self_attention', torch.zeros((1, 1, 0)).byte())
        self.register_state('running_seq', torch.zeros((1,)).long())

    def forward(self, t, input, encoder_output, mask_encoder, pos):
        """
        input: (B, L)
        encoder_output: (B, S, D)
        pos: positional enc for encoder_output, shape compatible with encoder_output
        """
        b_s, seq_len = input.shape[:2]
        mask_queries = (input != self.padding_idx).unsqueeze(-1).float()  # (B, L, 1)

        # causal mask
        mask_self_attention = torch.triu(
            torch.ones((seq_len, seq_len), dtype=torch.uint8, device=input.device), diagonal=1
        ).unsqueeze(0).unsqueeze(0)  # (1,1,L,L)
        mask_self_attention = mask_self_attention + (input == self.padding_idx).unsqueeze(1).unsqueeze(1).byte()
        mask_self_attention = mask_self_attention.gt(0)  # (B,1,L,L) 广播

        if self._is_stateful:
            self.running_mask_self_attention = torch.cat(
                [self.running_mask_self_attention.type_as(mask_self_attention), mask_self_attention], -1)
            mask_self_attention = self.running_mask_self_attention

        seq = torch.arange(1, seq_len + 1).view(1, -1).expand(b_s, -1).to(input.device)  # (B, L)
        seq = seq.masked_fill(mask_queries.squeeze(-1) == 0, 0)
        if self._is_stateful:
            self.running_seq.add_(1)
            seq = self.running_seq

        out = self.word_emb(input) + self.pos_emb(seq)  # (B,L,D)

        # beam-search特例（保持原实现）
        if encoder_output.shape[0] > pos.shape[0]:
            assert encoder_output.shape[0] % pos.shape[0] == 0
            beam_size = int(encoder_output.shape[0] / pos.shape[0])
            pos = pos.unsqueeze(1).expand(pos.shape[0], beam_size, pos.shape[1], pos.shape[2]).contiguous().flatten(0, 1)

        for l in self.layers:
            out = l(t, out, encoder_output, mask_queries, mask_self_attention, mask_encoder, pos=pos)

        out = self.fc(out)
        return F.log_softmax(out, dim=-1)
